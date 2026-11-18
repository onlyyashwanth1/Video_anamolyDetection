#!/usr/bin/env python3
"""Joint native-frame temporal ablations on fixed, shared window scorers.

This deliberately operates at the saved native frame rate. Temporal windows
are converted from seconds using each video's own fps; no labels influence
the interpolation, weighting, or smoothing. Every output includes all frames.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from search_xd_configs import load_run, window_scores, segment_scores, weighted_metrics, normalize_config
from search_shared_configs import RUNS

DESIGNS = ('original', 'nearest_center', 'linear_centers', 'triangular')
SMOOTHING = (0., 1., 2., 4., 8., 16., 32.)
SEEDS = {
    'new_prompt_shared': dict(action_temperature=.02, action_image_weight=.8125,
                              action_weight=1., pooling='mean',
                              image_prompt_variant='semantic_shared16_video',
                              text_prompt_variant='expanded26',
                              image_prompt_reducer='logmeanexp', text_prompt_reducer='max'),
    'joint_fusion': dict(action_temperature=.045, action_image_weight=.65,
                         action_weight=1., pooling='mean_max_half'),
    'xd_fusion': dict(action_temperature=.03, action_image_weight=.375,
                      action_weight=1., pooling='mean_max_half'),
    'original_bank_free': dict(action_temperature=.1, action_image_weight=.875,
                              action_weight=1., pooling='mean'),
    'ucf_minimax': dict(action_temperature=.01, action_image_weight=.75,
                        action_weight=1., pooling='mean'),
}


def load_temporal_run(path, split_path=None):
    cache = load_run(path) if split_path is None else load_run(path, split_path)
    records, window_offset = [], 0
    for index, name in enumerate(cache['names']):
        base = Path(path)/'predictions'/name
        data = json.loads(Path(str(base)+'.json').read_text())
        with np.load(str(base)+'.npz') as saved:
            y = saved['labels'].copy()
            fps = float(saved['fps'])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Native fps is required')
        starts = np.array([e['start_frame'] for e in data['events']], dtype=int)
        stops = np.array([e['stop_frame'] for e in data['events']], dtype=int)
        centers = (starts + stops - 1) / 2
        if np.any(np.diff(centers) <= 0):
            raise ValueError('Expected strictly increasing window centers')
        mask = cache['video_indices'] == index
        lengths = (cache['positive'][mask] + cache['negative'][mask]).astype(int)
        if lengths.sum() != len(y):
            raise ValueError('Native segment lengths do not reconstruct video')
        records.append(dict(name=name, labels=y, fps=fps, starts=starts, stops=stops,
                            centers=centers, frame_count=len(y), segment_mask=mask,
                            segment_lengths=lengths,
                            window_slice=slice(window_offset, window_offset+len(starts))))
        window_offset += len(starts)
    cache['temporal_records'] = records
    return cache


def temporal_scores(record, values, design, original=None, smooth_seconds=0.):
    """Return native-frame scores; replicated boundaries cover video tails."""
    if design not in DESIGNS:
        raise ValueError(design)
    n, centers = record['frame_count'], record['centers']
    frames = np.arange(n)
    if design == 'original':
        if original is None or len(original) != n:
            raise ValueError('Original native-frame scores required')
        scores = np.asarray(original, dtype=float)
    elif design == 'linear_centers':
        scores = np.interp(frames, centers, values)
    elif design == 'nearest_center':
        transitions = .5*(centers[:-1] + centers[1:])
        scores = np.asarray(values)[np.searchsorted(transitions, frames, side='right')]
    else:
        # Each window gets a triangular support over its exact saved interval.
        # The +1 ensures that first/last supported frame has nonzero weight.
        # Accumulate affine left/right pieces using difference arrays: O(n+k).
        num_slope, num_intercept = np.zeros(n+1), np.zeros(n+1)
        den_slope, den_intercept = np.zeros(n+1), np.zeros(n+1)
        starts, stops = record['starts'], record['stops']
        middle = np.ceil(centers).astype(int)
        half = (stops-starts+1)/2
        for start, stop, slope, intercept in (
            (starts, middle, 1/half, 1-centers/half),
            (middle, stops, -1/half, 1+centers/half),
        ):
            for array, coefficient in ((num_slope,slope*values), (num_intercept,intercept*values),
                                       (den_slope,slope), (den_intercept,intercept)):
                np.add.at(array, start, coefficient)
                np.add.at(array, stop, -coefficient)
        denominator = np.cumsum(den_slope[:-1])*frames + np.cumsum(den_intercept[:-1])
        numerator = np.cumsum(num_slope[:-1])*frames + np.cumsum(num_intercept[:-1])
        if np.any(denominator < 1e-9):
            raise ValueError('Triangular support left uncovered native frames')
        scores = numerator / denominator
    if smooth_seconds:
        if smooth_seconds < 0 or not np.isfinite(smooth_seconds):
            raise ValueError('Invalid smoothing width')
        # Odd boxcar width avoids a half-frame directional preference.
        width = max(1, int(round(smooth_seconds*record['fps'])))
        width += not width % 2
        scores = uniform_filter1d(scores, width, mode='nearest')
    if scores.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError('Invalid temporal output')
    return np.clip(scores, 0, 1)


def scoring_cache(cache, config):
    """Apply shared prompt image margins and exact independent probe scales."""
    change_prompt = config.get('image_prompt_variant') and cache.get('temporal_prompt_margins') is not None
    ti = config.get('image_temperature',config['action_temperature'])
    tt = config.get('text_temperature',config['action_temperature'])
    if not 0 < ti < np.inf or not 0 < tt < np.inf:
        raise ValueError('Both probe temperatures must be finite and positive')
    if abs(config['action_temperature']-ti) > 1e-12:
        raise ValueError('Legacy action_temperature must equal image_temperature')
    if change_prompt or ti != tt:
        features = cache['features'].copy()
        if change_prompt:
            key = (config['image_prompt_variant'], config['image_prompt_reducer'])
            features[:,0] = cache['temporal_prompt_margins'][key]
        # Existing helper evaluates both probes at Ti. Multiplying the raw
        # caption margin by Ti/Tt is exactly the independent Tt sigmoid.
        features[:,1] *= ti/tt
        cache = {**cache, 'features':features}
    return cache


def frame_outputs(cache, config, design='original', smooth_seconds=0., partition='validation'):
    config = normalize_config(config)
    cache = scoring_cache(cache,config)
    values = window_scores(cache, config)
    segments = segment_scores(cache, config)
    subset = set(cache['splits']['diagnostic_'+partition+'_videos']) if partition != 'all_benchmark' else set(cache['names'])
    ys, xs = [], []
    for record in cache['temporal_records']:
        if record['name'] not in subset:
            continue
        original = np.repeat(segments[record['segment_mask']], record['segment_lengths']) if design == 'original' else None
        scores = temporal_scores(record, values[record['window_slice']], design, original, smooth_seconds)
        ys.append(record['labels'])
        xs.append(scores)
    return np.concatenate(ys), np.concatenate(xs)


def evaluate_temporal(cache, config, design='original', smooth_seconds=0., partition='validation'):
    y, x = frame_outputs(cache, config, design, smooth_seconds, partition)
    return weighted_metrics(x, y, 1-y)


def run(caches, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start, rows = time.monotonic(), []
    # Reference the same original full-model validation objectives as root's
    # joint scoring search; no independent objective can favor one dataset.
    from search_xd_configs import evaluate, DEFAULT_CONFIG
    baseline = {name:evaluate(c, DEFAULT_CONFIG, 'validation') for name,c in caches.items()}
    u0, x0 = baseline['UCF-Crime']['roc_auc'], baseline['XD-Violence']['average_precision']
    for seed, config in SEEDS.items():
        for design in DESIGNS:
            for smooth in SMOOTHING:
                metrics = {name:evaluate_temporal(c, config, design, smooth) for name,c in caches.items()}
                if design == 'original' and smooth == 0:
                    for name,cache in caches.items():
                        if config.get('image_prompt_variant') and cache.get('temporal_prompt_margins') is not None:
                            features = cache['features'].copy()
                            features[:,0] = cache['temporal_prompt_margins'][(config['image_prompt_variant'],config['image_prompt_reducer'])]
                            cache = {**cache,'features':features}
                        reference = evaluate(cache, config, 'validation')
                        if any(abs(reference[m]-metrics[name][m]) > 1e-12 for m in reference):
                            raise ValueError('Native expansion differs from compressed baseline')
                gains = [metrics['UCF-Crime']['roc_auc']/u0-1, metrics['XD-Violence']['average_precision']/x0-1]
                row = dict(seed=seed, config=normalize_config(config), temporal_design=design,
                           smooth_seconds=smooth, validation=metrics,
                           joint_relative_gain=float(np.mean(gains)), minimum_relative_gain=float(min(gains)))
                rows.append(row)
                print(f'{len(rows)}/{len(SEEDS)*len(DESIGNS)*len(SMOOTHING)} {seed} {design} smooth={smooth}: UCF AUC={metrics["UCF-Crime"]["roc_auc"]:.5f}, XD AP={metrics["XD-Violence"]["average_precision"]:.5f}', flush=True)
    # Smoothing can shift the best visual/text fusion. One bounded common grid
    # refines the root's selected shared prompt recipe using validation only.
    seen = {(json.dumps(r['config'],sort_keys=True),r['temporal_design'],r['smooth_seconds']) for r in rows}
    for temp, image, pooling, smooth in itertools.product(
        [.01,.015,.02,.025,.03], [.75,.78125,.8125,.84375,.875],
        ['mean','mean_max_half'], [4.,8.,12.,16.]):
        config = normalize_config({**SEEDS['new_prompt_shared'], 'action_temperature':temp,
                                   'action_image_weight':image,'pooling':pooling})
        key = (json.dumps(config,sort_keys=True),'original',smooth)
        if key in seen:
            continue
        seen.add(key)
        metrics = {name:evaluate_temporal(c,config,'original',smooth) for name,c in caches.items()}
        gains = [metrics['UCF-Crime']['roc_auc']/u0-1,metrics['XD-Violence']['average_precision']/x0-1]
        rows.append(dict(seed='new_prompt_refinement',config=config,temporal_design='original',
                         smooth_seconds=smooth,validation=metrics,
                         joint_relative_gain=float(np.mean(gains)),minimum_relative_gain=float(min(gains))))
        if len(rows)%25 == 0:
            best = max(rows,key=lambda r:r['joint_relative_gain'])
            print(f'{len(rows)} candidates including common fusion/temporal refinement; best UCF={best["validation"]["UCF-Crime"]["roc_auc"]:.5f}, XD={best["validation"]["XD-Violence"]["average_precision"]:.5f}',flush=True)
    eligible = [r for r in rows if r['minimum_relative_gain'] >= 0]
    ranked = sorted(eligible or rows, key=lambda r:r['joint_relative_gain'], reverse=True)
    selected = ranked[0]
    shortlist = ranked[:10]
    best_ucf = max(rows,key=lambda r:r['validation']['UCF-Crime']['roc_auc'])
    best_xd = max(rows,key=lambda r:r['validation']['XD-Violence']['average_precision'])
    for row in (best_ucf,best_xd):
        if row not in shortlist:
            shortlist.append(row)
    # Always retain each seed's unchanged temporal design to isolate effects.
    for row in rows:
        if row['temporal_design'] == 'original' and row['smooth_seconds'] == 0 and row not in shortlist:
            shortlist.append(row)
    for row in shortlist:
        for partition in ('evaluation', 'all_benchmark'):
            row[partition] = {name:evaluate_temporal(c, row['config'], row['temporal_design'], row['smooth_seconds'], partition) for name,c in caches.items()}
    verification = {}
    for name,cache in caches.items():
        y,x = frame_outputs(cache, selected['config'], selected['temporal_design'], selected['smooth_seconds'], 'all_benchmark')
        verification[name] = dict(roc_auc_error=abs(roc_auc_score(y,x)-selected['all_benchmark'][name]['roc_auc']),
                                  ap_error=abs(average_precision_score(y,x)-selected['all_benchmark'][name]['average_precision']))
        if max(verification[name].values()) > 1e-12:
            raise ValueError('Selected metrics differ from independent expanded sklearn reference')
    result = dict(objective='Mean relative validation improvement in UCF ROC-AUC and XD AP; neither may regress if feasible.',
                  selected=selected, shortlist=shortlist, candidates=rows,
                  validation_only_best_ucf=best_ucf, validation_only_best_xd=best_xd,
                  candidate_count=len(rows), elapsed_seconds=time.monotonic()-start,
                  exact_expanded_metric_verification=verification,
                  provenance={name:dict(source_run=c['run_path'], source_summary_sha256=c['source_summary_sha256'],
                                        validation=c['splits']['diagnostic_validation_videos'], evaluation=c['splits']['diagnostic_evaluation_videos']) for name,c in caches.items()},
                  limitations=['Already inspected benchmarks: evaluation/all-benchmark remain reused-development estimates.',
                               'Fixed window evidence cannot recover unseen details from new frames.',
                               'All temporal methods use native frame indices and each clip fps; seconds/settings are shared.',
                               'Center interpolation, symmetric overlap and centered boxcar smoothing require future frames.',
                               'Boxcar width is the nearest odd native-frame count to the requested shared duration.',
                               'No memory-bank information, label-driven video normalization or threshold fitting.',
                               'new_prompt_shared uses the root-selected common semantic_shared16_video image prompts with logmeanexp(.05), and the untouched original expanded26 MAX text margins.'])
    (output/'search.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Shared temporal design experiments', '',
             f'{len(rows)} candidates; source-group validation selects the joint UCF ROC-AUC/XD AP objective. Evaluation/all values are descriptive reused-development estimates.', '',
             '| Seed | Temporal rule | Smooth seconds | Validation UCF AUC | Validation XD AP | All UCF AUC | All XD AP |', '|---|---|---:|---:|---:|---:|---:|']
    for row in shortlist:
        lines.append(f'| {row["seed"]} | {row["temporal_design"]} | {row["smooth_seconds"]:g} | '+
                     ' | '.join(f'{row[p][d][m]*100:.3f}' for p,d,m in [('validation','UCF-Crime','roc_auc'),('validation','XD-Violence','average_precision'),('all_benchmark','UCF-Crime','roc_auc'),('all_benchmark','XD-Violence','average_precision')])+' |')
    lines += ['', '## Limitations', ''] + ['- '+x for x in result['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved {output}/search.json in {time.monotonic()-start:.1f}s', flush=True)
    return result


def refine_temperatures(caches,output):
    """One final bounded grid, reusing prior validation rows without reruns."""
    from search_xd_configs import evaluate, DEFAULT_CONFIG
    from scipy.special import expit
    output = Path(output)
    result = json.loads((output/'search.json').read_text())
    rows = result['candidates']
    baseline = {name:evaluate(cache,DEFAULT_CONFIG,'validation') for name,cache in caches.items()}
    u0,x0 = baseline['UCF-Crime']['roc_auc'],baseline['XD-Violence']['average_precision']
    seen = {(json.dumps(r['config'],sort_keys=True),r['temporal_design'],r['smooth_seconds']) for r in rows}
    start = time.monotonic()
    tested = 0
    for ti,tt,image,smooth,pooling in itertools.product(
        [.015,.02,.025,.03],[.015,.02,.025,.03,.05],[.75,.8125,.875],[8.,12.,16.],['mean','mean_max_half']):
        config = normalize_config({**SEEDS['new_prompt_shared'],'action_temperature':ti,
                                   'image_temperature':ti,'text_temperature':tt,
                                   'action_image_weight':image,'pooling':pooling})
        key = (json.dumps(config,sort_keys=True),'original',smooth)
        if key in seen:
            continue
        seen.add(key)
        metrics = {name:evaluate_temporal(cache,config,'original',smooth) for name,cache in caches.items()}
        gains = [metrics['UCF-Crime']['roc_auc']/u0-1,metrics['XD-Violence']['average_precision']/x0-1]
        rows.append(dict(seed='final_independent_probe_temperatures',config=config,
                         temporal_design='original',smooth_seconds=smooth,validation=metrics,
                         joint_relative_gain=float(np.mean(gains)),minimum_relative_gain=float(min(gains))))
        tested += 1
        if tested%40 == 0:
            best = max(rows,key=lambda r:r['joint_relative_gain'])
            print(f'Final independent-temperature grid {tested}/360: best UCF={best["validation"]["UCF-Crime"]["roc_auc"]:.5f}, XD={best["validation"]["XD-Violence"]["average_precision"]:.5f}',flush=True)
    eligible = [r for r in rows if r['minimum_relative_gain']>=0]
    selected = max(eligible or rows,key=lambda r:r['joint_relative_gain'])
    best_ucf = max(rows,key=lambda r:r['validation']['UCF-Crime']['roc_auc'])
    best_xd = max(rows,key=lambda r:r['validation']['XD-Violence']['average_precision'])
    shortlist = sorted(eligible or rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:15]
    for row in (best_ucf,best_xd):
        if row not in shortlist:
            shortlist.append(row)
    for row in shortlist:
        for partition in ('evaluation','all_benchmark'):
            row[partition] = {name:evaluate_temporal(cache,row['config'],row['temporal_design'],row['smooth_seconds'],partition) for name,cache in caches.items()}
    verification = {}
    for name,cache in caches.items():
        y,x = frame_outputs(cache,selected['config'],selected['temporal_design'],selected['smooth_seconds'],'all_benchmark')
        error = dict(roc_auc_error=abs(roc_auc_score(y,x)-selected['all_benchmark'][name]['roc_auc']),
                     ap_error=abs(average_precision_score(y,x)-selected['all_benchmark'][name]['average_precision']))
        cfg=selected['config'];ti=cfg.get('image_temperature',cfg['action_temperature']);tt=cfg.get('text_temperature',cfg['action_temperature'])
        features=cache['features'].copy()
        features[:,0]=cache['temporal_prompt_margins'][(cfg['image_prompt_variant'],cfg['image_prompt_reducer'])]
        expected=cfg['action_image_weight']*expit(features[:,0]/ti)+(1-cfg['action_image_weight'])*expit(features[:,1]/tt)
        actual=window_scores(scoring_cache(cache,cfg),cfg)
        error['independent_probe_sigmoid_max_error']=float(np.max(np.abs(expected-actual)))
        if max(error.values()) > 1e-12:
            raise ValueError('Final temporal/probe-temperature verification failed')
        verification[name]=error
    result.update(selected=selected,shortlist=shortlist,candidates=rows,candidate_count=len(rows),
                  validation_only_best_ucf=best_ucf,validation_only_best_xd=best_xd,
                  exact_expanded_metric_verification=verification,
                  elapsed_seconds=result['elapsed_seconds']+time.monotonic()-start,
                  last_refinement=dict(candidates=tested,seconds=time.monotonic()-start,
                                       image_temperatures=[.015,.02,.025,.03],text_temperatures=[.015,.02,.025,.03,.05]))
    result['limitations'].append('Final independent probe temperatures are common across datasets; legacy action_temperature equals image_temperature, and original caption margin is rescaled only to implement the explicit text temperature exactly.')
    (output/'search.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Shared temporal and probe-temperature experiments','',
           f'{len(rows)} total recipes. Final bounded grid: {tested} independent image/text temperature recipes. Selection uses source-group validation only. Evaluation/all-benchmark are reused-development estimates.','',
           '| Choice | Image T | Text T | Image weight | Pooling | Smooth seconds | Validation UCF AUC | Validation XD AP | All UCF AUC | All XD AP |',
           '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for label,row in [('Joint validation choice',selected),('Best validation UCF',best_ucf),('Best validation XD',best_xd)]:
        cfg=row['config'];ti=cfg.get('image_temperature',cfg['action_temperature']);tt=cfg.get('text_temperature',cfg['action_temperature'])
        lines.append(f'| {label} | {ti} | {tt} | {cfg["action_image_weight"]} | {cfg["pooling"]} | {row["smooth_seconds"]} | '+
                     ' | '.join(f'{row[p][d][m]*100:.4f}' for p,d,m in [('validation','UCF-Crime','roc_auc'),('validation','XD-Violence','average_precision'),('all_benchmark','UCF-Crime','roc_auc'),('all_benchmark','XD-Violence','average_precision')])+' |')
    lines+=['','## Limitations','']+['- '+s for s in result['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected':selected,'validation_only_best_ucf':best_ucf,'validation_only_best_xd':best_xd},indent=2),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/results/shared_temporal_search_20261026')
    parser.add_argument('--refine-probe-temperatures',action='store_true')
    args = parser.parse_args()
    caches = {name:load_temporal_run(path) for name,path in RUNS.items()}
    from search_shared_prompts import margins
    feature_dir = ROOT/'outputs/results/shared_prompt_features_20261026'
    metadata = json.loads((feature_dir/'prompt_features.json').read_text())
    for name,short in [('UCF-Crime','ucf'),('XD-Violence','xd')]:
        with np.load(feature_dir/f'{short}_image_prompt_features.npz') as saved:
            if list(saved['names']) != caches[name]['names']:
                raise ValueError('Prompt feature video ordering differs')
            image_margin = margins(saved['image_similarities'], metadata['variants']['semantic_shared16_video'], metadata['prompt_ids'], 'logmeanexp')
            if len(image_margin) != len(caches[name]['features']):
                raise ValueError('Prompt feature window ordering differs')
        caches[name]['temporal_prompt_margins'] = {('semantic_shared16_video','logmeanexp'):image_margin}
    if args.refine_probe_temperatures:
        refine_temperatures(caches,args.output)
    else:
        run(caches, args.output)


if __name__ == '__main__':
    main()
