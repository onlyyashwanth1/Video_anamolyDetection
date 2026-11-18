#!/usr/bin/env python3
"""Reconstruct score ablations without inference; diagnostic grouped calibration."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from anomaly_detection.evaluation.dataset import expand_windows
from anomaly_detection.evaluation.metrics import compute_metrics

VARIANTS = {'visual_action_only': 'action_image_score', 'action_only': 'action_score',
            'memory_only': 'memory_score', 'full_model': None}


def analyze(folder):
    run = json.loads((folder / 'run.json').read_text())
    original = json.loads((folder / 'summary.json').read_text())
    if run['status'] != 'complete' or run['config']['temporal']['mode'] != 'sliding_windows':
        raise ValueError('Requires a completed sliding-window evaluation')
    names = sorted(original['per_video'])
    # XD movie/YouTube segments from the same source must stay together.
    groups = [n.split('__#')[0] if '__#' in n else n for n in names]
    val, test = next(GroupShuffleSplit(n_splits=1, train_size=.2, random_state=42).split(names, groups=groups))
    assert not ({groups[i] for i in val} & {groups[i] for i in test})
    records = []
    max_error = 0.
    weight = run['config']['inference']['action_weight']
    for name in names:
        path = folder / 'predictions' / name
        data = json.loads(Path(str(path)+'.json').read_text())
        with np.load(str(path)+'.npz') as saved:
            labels, baseline = saved['labels'].copy(), saved['scores'].copy()
        events = data['events']
        windows = [SimpleNamespace(start_frame=e['start_frame'], stop_frame=e['stop_frame']) for e in events]
        components = [e['components'] for e in events]
        scores = {}
        for variant, key in VARIANTS.items():
            values = [c[key] if key else weight*c['action_score']+(1-weight)*c['memory_score'] for c in components]
            scores[variant] = np.clip(expand_windows(windows, values, len(labels)), 0, 1)
        error = float(np.max(np.abs(scores['full_model']-baseline)))
        max_error = max(max_error, error)
        if error > 1e-9:
            raise ValueError(f'Reconstruction differs for {name}: {error}')
        records.append((labels, scores))
    def joined(indices, variant):
        return (np.concatenate([records[i][0] for i in indices]),
                np.concatenate([records[i][1][variant] for i in indices]))
    threshold = original['overall']['threshold']
    result = {'dataset':run['config']['evaluation']['dataset'], 'source_run':str(folder.resolve()),
              'source_summary_sha256':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
              'videos':len(names), 'max_full_reconstruction_error':max_error,
              'diagnostic_validation_videos':[names[i] for i in val],
              'diagnostic_evaluation_videos':[names[i] for i in test], 'variants':{}}
    for variant in VARIANTS:
        labels, scores = joined(range(len(names)), variant)
        vl, vs = joined(val, variant)
        tl, ts = joined(test, variant)
        if len(np.unique(vl)) != 2 or len(np.unique(tl)) != 2:
            raise ValueError('Diagnostic split requires both frame classes')
        precision, recall, thresholds = precision_recall_curve(vl, vs)
        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
        # compute_metrics uses strict >; include the selected score's tie group.
        selected = float(np.nextafter(thresholds[int(np.argmax(f1))], -np.inf))
        result['variants'][variant] = {
            'all_benchmark_frames_original_threshold':compute_metrics(labels, scores, threshold),
            'diagnostic_validation_selected_threshold':selected,
            'diagnostic_evaluation_original_threshold':compute_metrics(tl, ts, threshold),
            'diagnostic_evaluation_validation_threshold':compute_metrics(tl, ts, selected)}
    for metric in ('roc_auc', 'average_precision'):
        reconstructed = result['variants']['full_model']['all_benchmark_frames_original_threshold'][metric]
        if not np.isclose(reconstructed, original['overall'][metric], atol=1e-12, rtol=0):
            raise ValueError(f'Full-model {metric} does not match saved summary')
    result['limitations'] = [
        'Ablations reuse fixed embeddings, captions, prompts and normal-reference banks; they do not retrain or rebuild memory.',
        'Action-only includes visual and caption action evidence; visual-action-only excludes caption action evidence.',
        'Grouped 20% validation / 80% evaluation split uses seed 42; XD segments share their source video group.',
        'Calibration is exploratory within an already evaluated benchmark, not an independent test estimate.',
        'UCF prompts and fusion weights were previously developed using some UCF test videos.',
        'Original alert threshold is shared for comparability; calibrated variants each maximize validation frame F1.' ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [analyze(folder) for folder in args.runs]
    (args.output/'ablations.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    lines = ['# Saved-score ablations', '', 'All original benchmark frames; percentages. See ablations.json for split membership, provenance and limitations.', '',
             '| Dataset | Variant | AUC | AP | Precision | Recall | F1 |', '|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        for variant, value in result['variants'].items():
            m=value['all_benchmark_frames_original_threshold']; t=m['threshold_metrics']
            values=[m['roc_auc'],m['average_precision'],t['precision'],t['recall'],t['f1']]
            lines.append('| '+result['dataset']+' | '+variant+' | '+' | '.join(f'{v*100:.2f}' for v in values)+' |')
    lines += ['', '## Exploratory threshold calibration', '', 'Validation and evaluation contain separate source groups. These are reused benchmark data, not an independent test. Thresholds maximize validation F1.', '', '| Dataset | Variant | Threshold | Evaluation precision | Evaluation recall | Evaluation F1 | Original-threshold evaluation F1 |', '|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        for variant,value in result['variants'].items():
            t=value['diagnostic_evaluation_validation_threshold']['threshold_metrics']; old=value['diagnostic_evaluation_original_threshold']['threshold_metrics']
            lines.append(f"| {result['dataset']} | {variant} | {value['diagnostic_validation_selected_threshold']:.6f} | {t['precision']*100:.2f} | {t['recall']*100:.2f} | {t['f1']*100:.2f} | {old['f1']*100:.2f} |")
    lines += ['', '## Limitations', '']+['- '+s for s in results[0]['limitations']]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
