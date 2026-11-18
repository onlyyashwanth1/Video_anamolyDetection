#!/usr/bin/env python3
"""Jointly search one bank-free model for UCF AUC and XD AP using cached evidence."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from search_xd_configs import load_run, evaluate, segment_scores, weighted_metrics, normalize_config, DEFAULT_CONFIG, POOLING

RUNS={'UCF-Crime':ROOT/'outputs/ucf_expanded95_evaluation/20261022T192245Z',
      'XD-Violence':ROOT/'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z'}


def postprocess(cache, scores, mode='none', strength=0.):
    if mode=='none':return scores
    ids=cache['video_indices']; weights=cache['positive']+cache['negative']; count=len(cache['names'])
    means=np.bincount(ids,weights=scores*weights,minlength=count)/np.bincount(ids,weights=weights,minlength=count)
    if mode=='center':
        shifted=scores-strength*(means[ids]-.5)
        return expit((shifted-.5)/.1)
    if mode=='standardize':
        variance=np.bincount(ids,weights=(scores-means[ids])**2*weights,minlength=count)/np.bincount(ids,weights=weights,minlength=count)
        standardized=.5+.1*(scores-means[ids])/np.maximum(np.sqrt(variance[ids]),.01)
        shifted=(1-strength)*scores+strength*standardized
        return expit((shifted-.5)/.1)
    raise ValueError(mode)


def assess(cache, config, partition):
    scores=segment_scores(cache,config)
    scores=postprocess(cache,scores,config.get('video_normalization','none'),config.get('normalization_strength',0.))
    mask=cache['masks'][partition]
    return weighted_metrics(scores[mask],cache['positive'][mask],cache['negative'][mask])


def run(caches, output, initial=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline={name:evaluate(c,DEFAULT_CONFIG,'validation') for name,c in caches.items()}
    u0=baseline['UCF-Crime']['roc_auc']; x0=baseline['XD-Violence']['average_precision']
    rows=[];seen=set();start=time.monotonic()
    def score(candidate,stage):
        candidate={**normalize_config(candidate),**{k:candidate[k] for k in ('video_normalization','normalization_strength') if k in candidate}}
        if candidate['action_weight']!=1:return
        candidate['memory_image_weight']=1.;candidate['memory_scale']='calibrated'
        key=json.dumps(candidate,sort_keys=True)
        if key in seen:return
        seen.add(key)
        metrics={name:assess(c,candidate,'validation') for name,c in caches.items()}
        u=metrics['UCF-Crime']['roc_auc'];x=metrics['XD-Violence']['average_precision']
        relative=(u/u0-1,x/x0-1)
        rows.append({'config':candidate,'validation':metrics,'joint_relative_gain':float(np.mean(relative)),
                     'minimum_relative_gain':float(min(relative)), 'stage':stage})
        if len(rows)%200==0:
            print(f'{len(rows)} shared candidates in {time.monotonic()-start:.1f}s',flush=True)
    for temp,image,transform in itertools.product([.01,.02,.03,.05,.075,.1,.15,.2,.5,1.,10.],[0.,.125,.25,.375,.5,.625,.75,.875,1.],['before_fusion','after_fusion']):
        score(dict(action_weight=1.,action_temperature=temp,action_image_weight=image,action_transform=transform),'coarse_mean')
    if initial:
        for c in initial:score(c,'agent_candidates')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:25]
    seeds+=sorted(rows,key=lambda r:r['minimum_relative_gain'],reverse=True)[:10]
    for row,pooling in itertools.product(seeds,POOLING):score({**row['config'],'pooling':pooling},'pooling')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:15]
    for row,mode,strength in itertools.product(seeds,['center','standardize'],[.125,.25,.5,.75,1.]):
        score({**row['config'],'video_normalization':mode,'normalization_strength':strength},'label_free_video_normalization')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:12]
    for row,tempfactor,image_delta in itertools.product(seeds,[.6,.8,1,1.2,1.5],[-.1,-.05,0,.05,.1]):
        c=row['config'];score({**c,'action_temperature':float(c['action_temperature']*tempfactor),
                              'action_image_weight':float(np.clip(c['action_image_weight']+image_delta,0,1))},'refinement')
    eligible=[r for r in rows if r['minimum_relative_gain']>=0]
    selected=max(eligible or rows,key=lambda r:r['joint_relative_gain'])
    ranked=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)
    shortlist=ranked[:30]
    if selected not in shortlist:shortlist.append(selected)
    # Include vanilla baseline and bank-free counterpart as fixed comparators.
    comparators={name:{partition:evaluate(cache,DEFAULT_CONFIG,partition) for partition in ('validation','evaluation','all_benchmark')} for name,cache in caches.items()}
    for row in shortlist:
        for partition in ('evaluation','all_benchmark'):
            row[partition]={name:assess(c,row['config'],partition) for name,c in caches.items()}
    result={'objective':'Maximize mean relative validation improvement in UCF ROC-AUC and XD AP; require neither to regress if feasible.',
            'same_model':'One frozen CLIP/Gemma feature pipeline, common prompts and fusion/aggregation; action_weight=1 means no dataset-specific bank state.',
            'baseline':comparators,'selected':selected,'shortlist':shortlist,'candidates':rows,
            'candidate_count':len(rows),'elapsed_seconds':time.monotonic()-start,
            'validation_only_best_ucf':max(rows,key=lambda r:r['validation']['UCF-Crime']['roc_auc']),
            'validation_only_best_xd':max(rows,key=lambda r:r['validation']['XD-Violence']['average_precision']),
            'provenance':{name:{'source_run':c['run_path'],'source_summary_sha256':c['source_summary_sha256'],
                               'validation':c['splits']['diagnostic_validation_videos'],'evaluation':c['splits']['diagnostic_evaluation_videos']} for name,c in caches.items()},
            'limitations':['Already inspected benchmarks: evaluation/all-benchmark are development results, not independent tests.',
                           'Search uses validation only. One setting cannot necessarily attain each dataset\'s individual optimum.',
                           'Video normalization if selected uses the full unlabeled clip and is offline, not causal streaming.',
                           'No threshold search here; AUC/AP are threshold independent.']}
    (output/'search.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'count':len(rows),'selected':selected},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/results/shared_config_search_20261026')
    args=parser.parse_args()
    caches={name:load_run(path) for name,path in RUNS.items()}
    initial=[]
    for path in [ROOT/'outputs/results/xd_config_search_20261026/candidates.json']:
        if path.exists():initial+=json.loads(path.read_text())
    run(caches,args.output,initial)

if __name__=='__main__':main()
