#!/usr/bin/env python3
"""Exact frame-weighted offline scoring search; no inference or config mutation.

The public load_run/evaluate helpers also support UCF, so a caller can score
identical recipes on both datasets. Only the pre-existing validation groups are
used for search. Evaluation and full-benchmark numbers are descriptive because
these benchmark data have already been inspected.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / 'outputs/results/score_ablations_20261026/ablations.json'
DEFAULT_RUN = ROOT / 'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z'
DEFAULT_OUTPUT = ROOT / 'outputs/results/xd_config_search_20261026'
DEFAULT_CONFIG = dict(action_temperature=.1, action_image_weight=.875,
                      action_weight=.95, memory_image_weight=1.,
                      memory_scale='calibrated', action_transform='before_fusion',
                      pooling='mean')
POOLING = ('mean', 'max', 'top2', 'top3', 'median', 'q75', 'q90', 'mean_max_half')


def normalize_config(config):
    result = {**DEFAULT_CONFIG, **config}
    for key in ('action_temperature', 'action_image_weight', 'action_weight', 'memory_image_weight'):
        result[key] = round(float(result[key]), 10)
        if not np.isfinite(result[key]):
            raise ValueError(f'{key} must be finite')
    if result['action_temperature'] <= 0:
        raise ValueError('Temperature must be positive')
    for key in ('action_image_weight', 'action_weight', 'memory_image_weight'):
        if not 0 <= result[key] <= 1:
            raise ValueError(key)
    if result['memory_scale'] not in ('calibrated', 'raw'):
        raise ValueError('memory_scale')
    if result['action_transform'] not in ('before_fusion', 'after_fusion'):
        raise ValueError('action_transform')
    if result['pooling'] not in POOLING:
        raise ValueError('pooling')
    # Canonicalize parameters which have no effect, so inert memory settings
    # cannot occupy a bank-free shortlist repeatedly.
    if result['action_weight'] == 1:
        result['memory_image_weight'], result['memory_scale'] = 1., 'calibrated'
    elif result['action_weight'] == 0:
        result['action_temperature'], result['action_image_weight'] = .1, .875
        result['action_transform'] = 'before_fusion'
    return result


def config_key(config):
    return json.dumps(normalize_config(config), sort_keys=True, separators=(',', ':'))


def weighted_metrics(scores, positive, negative):
    """Exact frame AP/AUC using each constant-score segment's class counts."""
    order = np.argsort(scores, kind='stable')[::-1]
    x, p, n = scores[order], positive[order], negative[order]
    ends = np.r_[np.flatnonzero(np.diff(x)), len(x) - 1]
    tp, fp = np.cumsum(p)[ends], np.cumsum(n)[ends]
    if not tp[-1] or not fp[-1]:
        raise ValueError('Both frame classes required')
    tpr, fpr = np.r_[0., tp / tp[-1]], np.r_[0., fp / fp[-1]]
    # Step integration is exactly sklearn average_precision_score.
    ap = np.sum(np.diff(tpr) * tp / (tp + fp))
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * .5)
    return dict(roc_auc=float(auc), average_precision=float(ap))


def load_run(run_path=DEFAULT_RUN, split_path=DEFAULT_SPLITS, dataset=None):
    """Cache per-window components and boundary segments, never frame expansion."""
    run_path, split_path = Path(run_path), Path(split_path)
    summary = json.loads((run_path / 'summary.json').read_text())
    run = json.loads((run_path / 'run.json').read_text())
    dataset = dataset or run['config']['evaluation']['dataset']
    splits = next(x for x in json.loads(split_path.read_text()) if x['dataset'] == dataset)
    names = sorted(summary['per_video'])
    memberships = {n: 'validation' for n in splits['diagnostic_validation_videos']}
    memberships.update({n: 'evaluation' for n in splits['diagnostic_evaluation_videos']})
    if set(memberships) != set(names):
        raise ValueError('Split does not partition run')
    groups = lambda ns: {n.split('__#')[0] for n in ns}
    if groups(splits['diagnostic_validation_videos']) & groups(splits['diagnostic_evaluation_videos']):
        raise ValueError('Source groups overlap')
    fields = ('action_image_margin', 'action_text_margin', 'memory_image_score',
              'memory_text_score', 'memory_image_raw', 'memory_text_raw')
    features, segments, positive, negative, partitions, video_indices = [], [], [], [], [], []
    baseline_error = 0.
    original_scores, original_labels = [], []
    offset, max_active = 0, 0
    for video_index, name in enumerate(names):
        base = run_path / 'predictions' / name
        data = json.loads(Path(str(base) + '.json').read_text())
        with np.load(str(base) + '.npz') as saved:
            labels, baseline = saved['labels'].copy(), saved['scores'].copy()
        events = data['events']
        features.extend([[e['components'][f] for f in fields] for e in events])
        changes = {}
        for index, event in enumerate(events):
            start, stop = event['start_frame'], event['stop_frame']
            if not 0 <= start < stop <= len(labels):
                raise ValueError('Invalid exclusive frame window')
            changes.setdefault(start, []).append((index + offset, True))
            changes.setdefault(stop, []).append((index + offset, False))
        boundaries = sorted({0, len(labels), *changes})
        prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
        active = set()
        frame_reconstruction = np.zeros(len(labels))
        for start, stop in zip(boundaries, boundaries[1:]):
            for index, enter in changes.get(start, []):
                active.add(index) if enter else active.remove(index)
            if not active:
                raise ValueError('Uncovered frames')
            ids = sorted(active)
            max_active = max(max_active, len(ids))
            segments.append(ids)
            pos = int(prefix[stop] - prefix[start])
            positive.append(pos)
            negative.append(stop - start - pos)
            partitions.append(memberships[name] == 'validation')
            video_indices.append(video_index)
            frame_reconstruction[start:stop] = np.mean([events[i - offset]['score'] for i in ids])
        baseline_error = max(baseline_error, float(np.max(np.abs(frame_reconstruction - baseline))))
        original_scores.append(baseline)
        original_labels.append(labels)
        offset += len(events)
    ids = np.full((len(segments), max_active), -1, dtype=np.int64)
    for index, active in enumerate(segments):
        ids[index, :len(active)] = active
    count = (ids >= 0).sum(axis=1)
    val = np.asarray(partitions, dtype=bool)
    cache = dict(dataset=dataset, run_path=str(run_path.resolve()), summary=summary,
                 source_summary_sha256=hashlib.sha256((run_path / 'summary.json').read_bytes()).hexdigest(),
                 fields=fields, features=np.asarray(features), ids=ids, counts=count,
                 positive=np.asarray(positive, dtype=np.float64), negative=np.asarray(negative, dtype=np.float64),
                 masks=dict(validation=val, evaluation=~val, all_benchmark=np.ones(len(val), bool)),
                 video_indices=np.asarray(video_indices), names=names, splits=splits,
                 max_original_mean_reconstruction_error=baseline_error,
                 _window_scores={}, _pool_scores={})
    if baseline_error > 1e-9:
        raise ValueError(f'Original mean frame reconstruction error {baseline_error}')
    baseline = evaluate(cache, DEFAULT_CONFIG, 'all_benchmark')
    for metric in ('roc_auc', 'average_precision'):
        if not np.isclose(baseline[metric], summary['overall'][metric], rtol=0, atol=1e-12):
            raise ValueError(f'Weighted baseline {metric} differs: {baseline[metric]} vs {summary["overall"][metric]}')
    # Independently validate the weighted implementation against expanded frames.
    y, x = np.concatenate(original_labels), np.concatenate(original_scores)
    assert np.isclose(weighted_metrics(x, y, 1-y)['roc_auc'], roc_auc_score(y, x), atol=1e-12)
    assert np.isclose(weighted_metrics(x, y, 1-y)['average_precision'], average_precision_score(y, x), atol=1e-12)
    return cache


def window_scores(cache, config):
    c = normalize_config(config)
    feature = cache['features']
    ai, at, mi, mt, ri, rt = feature.T
    if c['action_transform'] == 'before_fusion':
        action = c['action_image_weight'] * expit(ai / c['action_temperature']) + (1-c['action_image_weight']) * expit(at / c['action_temperature'])
    else:
        action = expit((c['action_image_weight'] * ai + (1-c['action_image_weight']) * at) / c['action_temperature'])
    if c['memory_scale'] == 'raw':
        mi, mt = ri, rt
    memory = c['memory_image_weight'] * mi + (1-c['memory_image_weight']) * mt
    return c['action_weight'] * action + (1-c['action_weight']) * memory


def segment_scores(cache, config):
    c = normalize_config(config)
    values = window_scores(cache, c)
    ids, counts = cache['ids'], cache['counts']
    gathered = values[np.maximum(ids, 0)]
    gathered = np.where(ids >= 0, gathered, np.nan)
    method = c['pooling']
    if method == 'mean':
        return np.nansum(gathered, axis=1) / counts
    if method == 'max':
        return np.nanmax(gathered, axis=1)
    if method == 'mean_max_half':
        return .5 * (np.nansum(gathered, axis=1) / counts + np.nanmax(gathered, axis=1))
    ordered = np.sort(np.where(ids >= 0, gathered, -np.inf), axis=1)
    if method in ('top2', 'top3'):
        k = int(method[-1])
        return np.where(np.isfinite(ordered[:, -k:]), ordered[:, -k:], 0).sum(axis=1) / np.minimum(k, counts)
    q = dict(median=.5, q75=.75, q90=.9)[method]
    position = (counts-1) * q
    lower, upper = np.floor(position).astype(int), np.ceil(position).astype(int)
    pad = ids.shape[1] - counts
    lower, upper = ordered[np.arange(len(ids)), pad+lower], ordered[np.arange(len(ids)), pad+upper]
    return lower + (upper-lower) * (position - np.floor(position))


def evaluate(cache, config, partition='validation'):
    mask = cache['masks'][partition]
    scores = segment_scores(cache, config)
    return weighted_metrics(scores[mask], cache['positive'][mask], cache['negative'][mask])


def search(cache, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    start = time.monotonic()

    def score(config, stage):
        c = normalize_config(config)
        key = config_key(c)
        if key in seen:
            return
        seen.add(key)
        metrics = evaluate(cache, c)
        row = dict(config=c, validation=metrics, stage=stage)
        rows.append(row)
        if len(rows) % 100 == 0:
            best = max(rows, key=lambda r: r['validation']['average_precision'])
            print(f'{len(rows)} candidates in {time.monotonic()-start:.1f}s; best validation AP {best["validation"]["average_precision"]:.6f}', flush=True)

    score(DEFAULT_CONFIG, 'baseline')
    for temp, image, action, transform in itertools.product(
            [.03, .05, .1, .2, .5, 1.], [0., .25, .5, .625, .75, .875, 1.],
            [0., .25, .5, .75, .9, .95, 1.], ['before_fusion', 'after_fusion']):
        if transform == 'after_fusion' and image in (0., 1.):
            continue
        score(dict(action_temperature=temp, action_image_weight=image, action_weight=action,
                   action_transform=transform), 'coarse_action')
    seeds = sorted(rows, key=lambda r: r['validation']['average_precision'], reverse=True)[:20]
    # Include the original recipe even if validation AP places it below top 20.
    seeds += sorted([r for r in rows if r['config']['action_weight'] == 1],
                    key=lambda r: r['validation']['average_precision'], reverse=True)[:10]
    seeds += [rows[0]]
    for seed, memory_image, scale, pooling in itertools.product(seeds, [0., .5, 1.], ['calibrated', 'raw'], POOLING):
        score({**seed['config'], 'memory_image_weight':memory_image, 'memory_scale':scale, 'pooling':pooling}, 'memory_pooling')
    top = sorted(rows, key=lambda r: r['validation']['average_precision'], reverse=True)[:12]
    for seed in top:
        c = seed['config']
        temps = np.unique(np.clip(np.array([.7, .85, 1, 1.15, 1.3])*c['action_temperature'], .01, 2)).tolist()
        images = np.unique(np.clip(np.array([-.125, -.0625, 0, .0625, .125])+c['action_image_weight'], 0, 1)).tolist()
        actions = np.unique(np.clip(np.array([-.1, -.05, 0, .025, .05])+c['action_weight'], 0, 1)).tolist()
        for temp, image, action in itertools.product(temps, images, actions):
            score({**c, 'action_temperature':temp, 'action_image_weight':image, 'action_weight':action}, 'local_refinement')
    ranked = sorted(rows, key=lambda r: r['validation']['average_precision'], reverse=True)
    bank_free = [r for r in ranked if r['config']['action_weight'] == 1]
    # Validation-selected candidates, plus baseline. Do not rank by evaluation.
    shortlist = ranked[:40]
    for row in bank_free[:20]:
        if row not in shortlist:
            shortlist.append(row)
    if rows[0] not in shortlist:
        shortlist.append(rows[0])
    for row in shortlist:
        for partition in ('evaluation', 'all_benchmark'):
            row[partition] = evaluate(cache, row['config'], partition)
    result = dict(dataset=cache['dataset'], source_run=cache['run_path'],
                  source_summary_sha256=cache['source_summary_sha256'],
                  validation_videos=cache['splits']['diagnostic_validation_videos'],
                  evaluation_videos=cache['splits']['diagnostic_evaluation_videos'],
                  segments=len(cache['ids']), windows=len(cache['features']),
                  frames=int((cache['positive']+cache['negative']).sum()),
                  max_original_mean_reconstruction_error=cache['max_original_mean_reconstruction_error'],
                  elapsed_seconds=time.monotonic()-start, candidates=rows, validation_selected=ranked[0],
                  shortlist=shortlist, bank_free_shortlist=bank_free[:20], limitations=[
                      'All benchmark data have been previously inspected; evaluation and all-benchmark results are reused-development estimates.',
                      'Search selection uses only the existing source-disjoint validation split; no per-video fitting.',
                      'Recipe selection must jointly assess UCF ROC-AUC and XD AP using exactly identical settings.',
                      'Threshold selection cannot improve ROC-AUC or AP; no threshold is changed here.',
                      'Existing embeddings, captions, prompts and normal-reference memory are fixed.'])
    (output/'search.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'candidates.json').write_text(json.dumps([r['config'] for r in rows], indent=2)+'\n')
    with (output/'validation_candidates.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=[*DEFAULT_CONFIG, 'roc_auc', 'average_precision', 'stage'])
        w.writeheader()
        for row in ranked:
            w.writerow({**row['config'], **row['validation'], 'stage':row['stage']})
    lines = ['# XD saved-component configuration search', '',
             f'Searched {len(rows)} recipes using source-group validation only. All evaluation/benchmark figures remain reused-development estimates.', '',
             '| Recipe | Validation AP | Evaluation AP | All AP | All AUC |', '|---|---:|---:|---:|---:|']
    for row in shortlist[:15]+[rows[0]]:
        lines.append('| '+json.dumps(row['config'], sort_keys=True)+' | '+' | '.join(f'{row[p][m]*100:.3f}' for p,m in [('validation','average_precision'),('evaluation','average_precision'),('all_benchmark','average_precision'),('all_benchmark','roc_auc')])+' |')
    lines += ['', '## Limits', ''] + ['- '+x for x in result['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {output}/search.json; {len(rows)} candidates, {time.monotonic()-start:.1f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--splits', type=Path, default=DEFAULT_SPLITS)
    parser.add_argument('--dataset')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cache = load_run(args.run, args.splits, args.dataset)
    print(f'Loaded {cache["dataset"]}: {len(cache["features"])} windows, {len(cache["ids"])} segments; exact baseline checks passed', flush=True)
    search(cache, args.output)


if __name__ == '__main__':
    main()
