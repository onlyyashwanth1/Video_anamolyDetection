#!/usr/bin/env python3
"""Compare saved XD window aggregation, with thresholds selected on validation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from anomaly_detection.evaluation.metrics import compute_metrics
from anomaly_detection.evaluation.window_aggregation import METHODS, aggregate_windows


def best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Evaluation uses strict >; include all ties at the selected score.
    return float(np.nextafter(thresholds[np.argmax(f1)], -np.inf))


def with_diagnostics(labels, scores, threshold, names, lengths, categories):
    result = compute_metrics(labels, scores, threshold)
    t = result['threshold_metrics']
    t['false_positive_rate'] = t['fp'] / (t['fp'] + t['tn'])
    missed = normal_alerts = normal_fp = normal_frames = 0
    offset = 0
    for name, length in zip(names, lengths):
        y = labels[offset:offset + length].astype(bool)
        decision = scores[offset:offset + length] > threshold
        if y.any():
            missed += not np.any(decision & y)
        if categories[name] == 'Normal':
            normal_alerts += decision.any()
            normal_fp += int(decision.sum())
            normal_frames += length
        offset += length
    result.update(anomalous_clips_completely_missed=int(missed),
                  normal_clips_with_alert=int(normal_alerts),
                  normal_clip_frame_false_positive_rate=normal_fp / normal_frames)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z')
    parser.add_argument('--splits', type=Path, default=ROOT / 'outputs/results/score_ablations_20261026/ablations.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/results/xd_aggregation_comparison_20261026')
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    run = json.loads((args.run / 'run.json').read_text())
    if run['status'] != 'complete' or run['config']['temporal']['mode'] != 'sliding_windows':
        raise ValueError('Requires a completed sliding-window run')
    split = next(d for d in json.loads(args.splits.read_text()) if d['dataset'] == 'XD-Violence')
    names = sorted(summary['per_video'])
    validation, evaluation = split['diagnostic_validation_videos'], split['diagnostic_evaluation_videos']
    if set(validation) & set(evaluation) or set(validation) | set(evaluation) != set(names):
        raise ValueError('Split does not partition this run')
    if ({n.split('__#')[0] for n in validation} & {n.split('__#')[0] for n in evaluation}):
        raise ValueError('Source-video groups overlap')
    labels, lengths = {}, {}
    scores = {method: {} for method in METHODS}
    max_error = 0.
    for index, name in enumerate(names, 1):
        base = args.run / 'predictions' / name
        data = json.loads(Path(str(base) + '.json').read_text())
        with np.load(str(base) + '.npz') as saved:
            labels[name] = saved['labels'].copy()
            baseline = saved['scores'].copy()
        lengths[name] = len(baseline)
        events = [SimpleNamespace(start_frame=e['start_frame'], stop_frame=e['stop_frame']) for e in data['events']]
        pooled = aggregate_windows(events, [e['score'] for e in data['events']], len(baseline))
        max_error = max(max_error, float(np.max(np.abs(pooled['mean'] - baseline))))
        for method in METHODS:
            scores[method][name] = pooled[method]
        if index % 100 == 0:
            print(f'Reconstructed {index}/{len(names)} videos', flush=True)
    if max_error > 1e-9:
        raise ValueError(f'Mean aggregation differs from original: {max_error}')
    categories = {n: m['category'] for n, m in summary['per_video'].items()}
    original_threshold = summary['overall']['threshold']
    result = dict(source_run=str(args.run.resolve()), original_threshold=original_threshold,
                  source_summary_sha256=hashlib.sha256((args.run / 'summary.json').read_bytes()).hexdigest(),
                  validation_videos=validation, evaluation_videos=evaluation,
                  max_mean_reconstruction_error=max_error, methods={})
    row_list = []
    for method in METHODS:
        metrics = {}
        validation_labels = np.concatenate([labels[n] for n in validation])
        validation_scores = np.concatenate([scores[method][n] for n in validation])
        threshold = best_f1_threshold(validation_labels, validation_scores)
        for partition, subset in [('validation', validation), ('evaluation', evaluation), ('all_benchmark', names)]:
            y = np.concatenate([labels[n] for n in subset])
            x = np.concatenate([scores[method][n] for n in subset])
            sizes = [lengths[n] for n in subset]
            for setting, cutoff in [('original_threshold', original_threshold), ('validation_threshold', threshold)]:
                key = partition + '_' + setting
                m = with_diagnostics(y, x, cutoff, subset, sizes, categories)
                metrics[key] = m
                t = m['threshold_metrics']
                row_list.append(dict(method=method, partition=partition, threshold_setting=setting,
                                     threshold=cutoff, auc=m['roc_auc'], ap=m['average_precision'],
                                     precision=t['precision'], recall=t['recall'], f1=t['f1'],
                                     fpr=t['false_positive_rate'],
                                     missed_clips=m['anomalous_clips_completely_missed'],
                                     normal_clips_with_alert=m['normal_clips_with_alert']))
        result['methods'][method] = metrics
        print(f'{method}: validation AP={metrics["validation_validation_threshold"]["average_precision"]:.5f}; '
              f'evaluation AP={metrics["evaluation_validation_threshold"]["average_precision"]:.5f}; '
              f'evaluation F1={metrics["evaluation_validation_threshold"]["threshold_metrics"]["f1"]:.5f}', flush=True)
    selected_ap = max(METHODS, key=lambda m: result['methods'][m]['validation_validation_threshold']['average_precision'])
    selected_f1 = max(METHODS, key=lambda m: result['methods'][m]['validation_validation_threshold']['threshold_metrics']['f1'])
    result['validation_selected_methods'] = dict(average_precision=selected_ap, f1=selected_f1)
    result['limitations'] = [
        'Exploratory comparison on an already inspected benchmark, not independent testing.',
        '170 validation / 630 evaluation videos; source groups are disjoint and unchanged from the earlier ablation.',
        'Each threshold maximizes its method\'s validation frame F1. AP and AUC do not depend on threshold.',
        'Validation determines recommended methods. Evaluation rankings and full-benchmark metrics are descriptive.',
        'All methods reuse the same fixed window scores; inference, captions, prompts and deployed settings are unchanged.',
        'Top-k uses the k highest covering window scores, or all scores when fewer than k cover a frame. Quantiles use linear interpolation.'
    ]
    for metric in ('roc_auc', 'average_precision'):
        if not np.isclose(result['methods']['mean']['all_benchmark_original_threshold'][metric], summary['overall'][metric], atol=1e-12, rtol=0):
            raise ValueError(f'Original mean {metric} does not match')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]))
        writer.writeheader()
        writer.writerows(row_list)
    lines = ['# XD window aggregation comparison', '',
             f'Validation-selected methods: **{selected_ap} for AP**, **{selected_f1} for F1**.', '',
             'Exploratory source-group split: 170 validation videos, 630 evaluation videos. Thresholds selected only on validation. All scores below are percentages except thresholds and clip counts.', '',
             '| Method | Validation AP | Validation F1 | Threshold | Evaluation AP | Evaluation AUC | Precision | Recall | F1 | FPR | Missed clips | Normal clips with alerts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, ms in result['methods'].items():
        v, e = ms['validation_validation_threshold'], ms['evaluation_validation_threshold']
        t = e['threshold_metrics']
        lines.append(f'| {method} | {v["average_precision"]*100:.2f} | {v["threshold_metrics"]["f1"]*100:.2f} | {e["threshold"]:.6f} | {e["average_precision"]*100:.2f} | {e["roc_auc"]*100:.2f} | {t["precision"]*100:.2f} | {t["recall"]*100:.2f} | {t["f1"]*100:.2f} | {t["false_positive_rate"]*100:.2f} | {e["anomalous_clips_completely_missed"]} | {e["normal_clips_with_alert"]} |')
    lines += ['', '## Full benchmark, descriptive only', '',
              '| Method | AP | AUC | Original-threshold F1 | Validation-threshold F1 |', '|---|---:|---:|---:|---:|']
    for method, ms in result['methods'].items():
        a, b = ms['all_benchmark_original_threshold'], ms['all_benchmark_validation_threshold']
        lines.append(f'| {method} | {a["average_precision"]*100:.2f} | {a["roc_auc"]*100:.2f} | {a["threshold_metrics"]["f1"]*100:.2f} | {b["threshold_metrics"]["f1"]*100:.2f} |')
    best_eval_ap = max(METHODS, key=lambda m: result['methods'][m]['evaluation_validation_threshold']['average_precision'])
    best_eval_f1 = max(METHODS, key=lambda m: result['methods'][m]['evaluation_validation_threshold']['threshold_metrics']['f1'])
    selected = result['methods'][selected_f1]['evaluation_validation_threshold']
    baseline = result['methods']['mean']['evaluation_original_threshold']
    calibrated_mean = result['methods']['mean']['evaluation_validation_threshold']
    normal_count = sum(categories[n] == 'Normal' for n in evaluation)
    lines += ['', '## Interpretation', '',
              f'Validation selects {selected_f1}, at threshold {selected["threshold"]:.6f}. '
              f'On evaluation, its F1 is {selected["threshold_metrics"]["f1"]*100:.2f}%, '
              f'compared with {baseline["threshold_metrics"]["f1"]*100:.2f}% for original mean pooling '
              f'and {calibrated_mean["threshold_metrics"]["f1"]*100:.2f}% for calibrated mean pooling. '
              'Most improvement comes from threshold calibration; aggregation adds a small further gain.', '',
              f'The selected setting alerts on {selected["normal_clips_with_alert"]}/{normal_count} normal evaluation clips. '
              f'Its normal-video frame false-positive rate is {selected["normal_clip_frame_false_positive_rate"]*100:.2f}%. '
              'This is a substantial alert burden despite the improved F1.', '',
              f'The highest observed evaluation AP is from {best_eval_ap}, and evaluation F1 from {best_eval_f1}. '
              'These evaluation winners are descriptive and do not replace validation selection. '
              'Small differences between methods have not been established as statistically reliable. '
              'The reused benchmark and prior inspection also limit any generalization claim.', '',
              'Mean reconstruction matches all saved frame scores within 1e-9; baseline pooled AUC/AP match within 1e-12. '
              'No deployed model or threshold settings were changed.']
    lines += ['', '## Limitations', ''] + ['- ' + s for s in result['limitations']]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Saved {args.output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
