#!/usr/bin/env python3
"""Audit saved XD scores; temporal and caption findings are diagnostic proxies."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from anomaly_detection.evaluation.dataset import expand_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z'))
    p.add_argument('--output', type=Path, default=Path('outputs/results/xd_missed_audit_20261026'))
    args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    summary=json.loads((args.run/'summary.json').read_text())
    calibration=json.loads(Path('outputs/results/score_ablations_20261026/ablations.json').read_text())
    diagnostic=next(d for d in calibration if d['dataset']=='XD-Violence')
    lower=diagnostic['variants']['full_model']['diagnostic_validation_selected_threshold']
    threshold=summary['overall']['threshold']; rows=[]; examples=[]; intervals=[]
    averaging={'missed_with_above_threshold_overlapping_window':0,'missed_with_above_threshold_half_anomalous_window':0}
    for name,m in summary['per_video'].items():
        if m['category']=='Normal': continue
        base=args.run/'predictions'/name
        d=json.loads(Path(str(base)+'.json').read_text())
        with np.load(str(base)+'.npz') as a: labels=a['labels'].astype(bool); scores=a['scores']
        events=d['events']; windows=[SimpleNamespace(start_frame=e['start_frame'],stop_frame=e['stop_frame']) for e in events]
        def expand(key):
            return expand_windows(windows,[e['components'][key] for e in events],len(labels))
        visual,action=expand('action_image_score'),expand('action_score')
        edges=np.diff(np.r_[False,labels,False].astype(int)); starts=np.where(edges==1)[0]; stops=np.where(edges==-1)[0]
        durations=(stops-starts)/d['fps']; missed=m['threshold_metrics']['tp']==0
        ranks=m['roc_auc']; informative=ranks is not None
        bucket=('detected_some_frames' if not missed else 'rescued_at_diagnostic_threshold' if np.any(scores[labels]>lower)
                else 'below_diagnostic_threshold')
        row={'video':name,'labels':name.split('_label_')[-1], 'missed':missed,'bucket':bucket,
             'anomaly_seconds':float(labels.sum()/d['fps']), 'longest_interval_seconds':float(max(durations)),
             'anomaly_score_max':float(scores[labels].max()),'anomaly_score_mean':float(scores[labels].mean()),
             'normal_score_mean':float(scores[~labels].mean()) if (~labels).any() else None,
             'auc':ranks,'ap':m['average_precision'],'recall':m['threshold_metrics']['recall'],
             'diagnostic_recall':float((scores[labels]>lower).mean()),
             'visual_anomaly_score_mean':float(visual[labels].mean()),'action_anomaly_score_mean':float(action[labels].mean()),
             'all_intervals_under_4s':bool(max(durations)<4), 'good_within_clip_ranking':bool(informative and ranks>=.8)}
        rows.append(row)
        for start,stop in zip(starts,stops):
            region=scores[start:stop]
            intervals.append({'video':name,'seconds':float((stop-start)/d['fps']),
                              'detected':bool(np.any(region>threshold)), 'clip_missed':missed})
        if missed:
            relevant=[e for e in events if labels[e['start_frame']:e['stop_frame']].any() and e['score']>threshold]
            averaging['missed_with_above_threshold_overlapping_window']+=bool(relevant)
            averaging['missed_with_above_threshold_half_anomalous_window']+=any(labels[e['start_frame']:e['stop_frame']].mean()>=.5 for e in relevant)
            # Choose window with greatest fraction of its frames annotated anomalous.
            overlap=[max(0,int(labels[e['start_frame']:e['stop_frame']].sum()))/(e['stop_frame']-e['start_frame']) for e in events]
            idx=int(np.argmax(overlap)); e=events[idx]
            examples.append({'video':name,'bucket':bucket,'auc':ranks,'anomaly_score_max':row['anomaly_score_max'],
                'window_start_seconds':e['start_time'],'annotated_fraction':overlap[idx],
                'caption':e['caption'],'components':e['components']})
    missed=[r for r in rows if r['missed']]
    counts=Counter(r['bucket'] for r in rows)
    duration_stats={}
    for lo,hi,title in [(0,1,'under_1s'),(1,4,'1_to_4s'),(4,10,'4_to_10s'),(10,float('inf'),'10s_or_longer')]:
        group=[r for r in intervals if lo<=r['seconds']<hi]
        duration_stats[title]={'intervals':len(group),'intervals_with_any_detection':sum(r['detected'] for r in group)}
    codes={}
    for code in sorted({c for r in rows for c in r['labels'].split('-') if c!='0'}):
        group=[r for r in rows if code in r['labels'].split('-')]
        codes[code]={'clips':len(group),'missed':sum(r['missed'] for r in group)}
    report={'threshold':threshold,'diagnostic_threshold':lower,'counts':dict(counts),
        'missed_clips_with_auc_at_least_0_8':sum(r['good_within_clip_ranking'] for r in missed),
        'missed_clips_with_auc_below_0_5':sum(r['auc'] is not None and r['auc']<.5 for r in missed),
        'missed_clips_all_intervals_under_4s':sum(r['all_intervals_under_4s'] for r in missed),
        'missed_clips_longest_interval_at_least_10s':sum(r['longest_interval_seconds']>=10 for r in missed),
        'window_averaging_diagnostics':averaging, 'interval_duration_stats':duration_stats,'label_code_stats_nonexclusive':codes,
        'limitations':['Saved-caption inspection is not a visual ground-truth review.',
        'Low scores do not establish whether the cause is sampling, captioning, or prompt scoring.',
        'The lower threshold comes from exploratory benchmark validation; applying it across all clips is descriptive, not independent testing.',
        'Interval detection means at least one detected frame, not complete localization. Label codes are nonexclusive.']}
    with (args.output/'clips.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'missed_caption_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    for bucket in ('rescued_at_diagnostic_threshold','below_diagnostic_threshold'):
        group=[e for e in examples if e['bucket']==bucket]
        group.sort(key=lambda e: e['auc'] if e['auc'] is not None else -1,reverse=True)
        for e in group[:4]: print(json.dumps(e))

if __name__=='__main__': main()
