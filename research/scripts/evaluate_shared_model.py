#!/usr/bin/env python3
"""Evaluate one bank-free configuration through production scoring on saved videos.

This intentionally does not import any hyperparameter search code. Native-frame
labels and video identities come from the original completed evaluations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from anomaly_detection.evaluation.metrics import compute_metrics, save_curves
from anomaly_detection.inference.action_scoring import ActionScorer
from anomaly_detection.inference.frame_projection import project_window_scores

DEFAULT_FEATURES = ROOT / 'outputs/results/shared_prompt_features_20261026'
DEFAULT_RUNS = [ROOT / 'outputs/ucf_expanded95_evaluation/20261022T192245Z',
                ROOT / 'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z']
DEFAULT_SPLITS = ROOT / 'outputs/results/score_ablations_20261026/ablations.json'
DATASET_KEYS = {'UCF-Crime': 'ucf', 'XD-Violence': 'xd'}


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(data)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def feature_contract(config):
    temporal = config.get('temporal', {})
    captioning = config.get('captioning', {})
    reasoning = config.get('reasoning', {})
    return dict(encoder=config['encoder']['model_name'], target_fps=config['video']['target_fps'],
                temporal={k: temporal.get(k) for k in ('mode', 'window_seconds', 'stride_seconds')},
                captioning={k: captioning.get(k) for k in ('mode', 'max_frames_per_window')},
                caption_model={k: reasoning.get(k) for k in ('backend', 'model', 'model_digest', 'quantization')})


class CachedTextEncoder:
    """Exact cached prompt vectors, with an offline text-only fallback."""

    def __init__(self, features):
        metadata = json.loads((features / 'prompt_features.json').read_text())
        vectors = np.load(features / 'prompt_embeddings.npy', mmap_mode='r')
        self.model_name = metadata['model']
        self.lookup = {text: vectors[i] for i, text in enumerate(metadata['prompts'])}
        self.model = self.tokenizer = None

    def encode_text(self, text):
        if text not in self.lookup:
            import torch
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
            if self.model is None:
                torch.set_num_threads(4)
                self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name, local_files_only=True)
                self.model = CLIPTextModelWithProjection.from_pretrained(
                    self.model_name, local_files_only=True).eval()
            inputs = self.tokenizer([text], padding=True, return_tensors='pt')
            if inputs['input_ids'].shape[1] > 77:
                raise ValueError('Prompt exceeds the baseline CLIP text context')
            with torch.inference_mode():
                vector = self.model(**inputs).text_embeds
                self.lookup[text] = (vector / vector.norm(dim=-1, keepdim=True))[0].cpu().numpy()
        return self.lookup[text]


def evaluate_source(source, output, config, config_sha, model_sha, features,
                    scorer, caption_lookup, caption_vectors, split_records, curves=False):
    source = Path(source).resolve()
    source_run = json.loads((source / 'run.json').read_text())
    if source_run['status'] != 'complete':
        raise ValueError(f'Source run incomplete: {source}')
    dataset = source_run['config']['evaluation']['dataset']
    if dataset not in DATASET_KEYS:
        raise ValueError(f'No prepared cache mapping for dataset {dataset}')
    key = DATASET_KEYS[dataset]
    if feature_contract(config) != feature_contract(source_run['config']):
        raise ValueError(f'Feature encoder/sampling/caption contract differs from source {dataset}')
    original = json.loads((source / 'summary.json').read_text())
    with np.load(features / (key + '_windows.npz'), allow_pickle=False) as z:
        windows = {k: z[k] for k in z.files}
    names, offsets = windows['names'].tolist(), windows['offsets']
    if names != sorted(original['per_video']) or offsets[0] != 0 or offsets[-1] != len(windows['images']):
        raise ValueError('Prepared windows differ from source video manifest')
    images = windows['images']
    if not np.isfinite(images).all() or not np.allclose(np.linalg.norm(images, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid cached image embeddings')
    try:
        text_rows = np.asarray([caption_lookup[c] for c in windows['captions']])
    except KeyError as error:
        raise ValueError('A source caption is missing from the exact-string text cache') from error
    texts = np.asarray(caption_vectors[text_rows])
    if not np.isfinite(texts).all() or not np.allclose(np.linalg.norm(texts, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid cached caption embeddings')
    # This is the actual production scorer, independent of the search's margin
    # reduction and sigmoid implementations.
    components = scorer.score_vectors(images, texts)
    scores_windows = np.asarray(components['action_score'], dtype=float)
    if scores_windows.shape != (len(images),):
        raise ValueError('Production scorer returned an invalid window score shape')
    split = next(r for r in split_records if r['dataset'] == dataset)
    validation = set(split['diagnostic_validation_videos'])
    evaluation = set(split['diagnostic_evaluation_videos'])
    group = lambda n: n.split('__#')[0]
    if validation & evaluation or validation | evaluation != set(names):
        raise ValueError('Diagnostic partitions do not match source video manifest')
    if {group(n) for n in validation} & {group(n) for n in evaluation}:
        raise ValueError('Diagnostic partitions overlap source-video groups')
    folder = output / key
    predictions = folder / 'predictions'
    predictions.mkdir(parents=True)
    threshold = float(config['inference']['anomaly_threshold'])
    records, per_video = [], {}
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    config=config, config_sha256=config_sha, model_sha256=model_sha,
                    dataset=dataset, source_run=str(source),
                    source_summary_sha256=digest_file(source / 'summary.json'),
                    feature_contract=feature_contract(config), feature_cache=str(features.resolve()),
                    evaluation_grid='all original video frames', normal_reference_memory_used=False,
                    reference_protocol='one shared bank-free scoring model for both datasets',
                    score_projection=config['temporal'], threshold_selected_during_this_evaluation=False,
                    validation_videos=sorted(validation), evaluation_videos=sorted(evaluation),
                    limitations=['Previously inspected benchmark data; these are development results.',
                                 'Centered smoothing, if configured, uses future context for offline evaluation.'])
    write_json(folder / 'run.json', metadata)
    for index, name in enumerate(names):
        a, b = map(int, offsets[index:index + 2])
        source_json = json.loads((source / 'predictions' / (name + '.json')).read_text())
        events = source_json['events']
        if len(events) != b - a:
            raise ValueError(f'Window count mismatch for {name}')
        for row, event in enumerate(events, a):
            if (int(windows['start_frame'][row]), int(windows['stop_frame'][row]),
                    windows['captions'][row]) != (event['start_frame'], event['stop_frame'], event['caption']):
                raise ValueError(f'Window/caption identity mismatch for {name}')
        with np.load(source / 'predictions' / (name + '.npz'), allow_pickle=False) as z:
            labels, fps = z['labels'].copy(), float(z['fps'])
        if fps != source_json['fps'] or len(labels) != source_json['frame_count']:
            raise ValueError(f'Native frame metadata mismatch for {name}')
        event_objects = [SimpleNamespace(start_frame=e['start_frame'], stop_frame=e['stop_frame'])
                         for e in events]
        scores = project_window_scores(event_objects, scores_windows[a:b], len(labels), fps,
                                       config['temporal'])
        metrics = compute_metrics(labels, scores, threshold)
        per_video[name] = dict(category=source_json['category'], partition=(
            'validation' if name in validation else 'evaluation'), **metrics)
        np.savez_compressed(predictions / (name + '.npz'), scores=scores, labels=labels,
                            final_decisions=(scores > threshold).astype(np.uint8), fps=np.asarray(fps),
                            window_scores=scores_windows[a:b], start_frame=windows['start_frame'][a:b],
                            stop_frame=windows['stop_frame'][a:b])
        records.append((name, labels, scores))
        if (index + 1) % 100 == 0:
            print(f'{dataset}: projected {index+1}/{len(names)} videos', flush=True)
    def pooled(subset):
        chosen = [r for r in records if r[0] in subset]
        return compute_metrics(np.concatenate([r[1] for r in chosen]),
                               np.concatenate([r[2] for r in chosen]), threshold)
    summary = dict(dataset=dataset, videos_evaluated=len(names), config_sha256=config_sha,
                   model_sha256=model_sha, overall=pooled(set(names)),
                   partitions=dict(validation=pooled(validation), evaluation=pooled(evaluation)),
                   normal_videos=original['normal_videos'], anomalous_videos=original['anomalous_videos'],
                   normal_reference_memory_used=False, per_video=per_video)
    write_json(folder / 'summary.json', summary)
    if curves:
        save_curves(np.concatenate([r[1] for r in records]),
                    np.concatenate([r[2] for r in records]), folder, dataset)
    metadata.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder / 'run.json', metadata)
    print(f'{dataset}: AUC={summary["overall"]["roc_auc"]:.8f}, '
          f'AP={summary["overall"]["average_precision"]:.8f}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-run', action='append', type=Path)
    parser.add_argument('--feature-cache', type=Path, default=DEFAULT_FEATURES)
    parser.add_argument('--split-file', type=Path, default=DEFAULT_SPLITS)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--save-curves', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['inference']['scoring_mode'] != 'action':
        raise ValueError('This evaluator requires bank-free action scoring')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_sha = digest_file(args.config)
    model_sha = hashlib.sha256(json.dumps(dict(features=feature_contract(config),
        inference=config['inference'], temporal=config['temporal']), sort_keys=True).encode()).hexdigest()
    encoder = CachedTextEncoder(args.feature_cache)
    if encoder.model_name != config['encoder']['model_name']:
        raise ValueError('Prompt embeddings use a different CLIP model')
    scorer = ActionScorer(encoder, config['inference'].get('action', {}))
    caption_texts = json.loads((args.feature_cache / 'caption_texts.json').read_text())
    caption_lookup = {text: i for i, text in enumerate(caption_texts)}
    caption_vectors = np.load(args.feature_cache / 'caption_embeddings.npy', mmap_mode='r')
    progress = json.loads((args.feature_cache / 'caption_progress.json').read_text())
    caption_sha = hashlib.sha256('\n'.join(caption_texts).encode()).hexdigest()
    if (progress['completed'] != len(caption_texts) or progress['texts_sha256'] != caption_sha
            or caption_vectors.shape != (len(caption_texts), 768)):
        raise ValueError('Caption embedding cache is incomplete')
    split_records = json.loads(args.split_file.read_text())
    sources = args.source_run or DEFAULT_RUNS
    metadata = dict(status='running', config=config, config_path=str(args.config.resolve()),
                    config_sha256=config_sha, model_sha256=model_sha,
                    source_runs=[str(s.resolve()) for s in sources],
                    production_scorer='ActionScorer.score_vectors + project_window_scores',
                    normal_reference_memory_used=False)
    write_json(args.output_dir / 'run.json', metadata)
    summaries = {}
    for source in sources:
        summary = evaluate_source(source, args.output_dir, config, config_sha, model_sha,
                                  args.feature_cache, scorer, caption_lookup, caption_vectors,
                                  split_records, args.save_curves)
        if summary['dataset'] in summaries:
            raise ValueError('Duplicate source dataset')
        summaries[summary['dataset']] = summary
    compact = {dataset: {k: v for k, v in summary.items() if k != 'per_video'}
               for dataset, summary in summaries.items()}
    write_json(args.output_dir / 'summary.json', dict(config_sha256=config_sha,
               model_sha256=model_sha, datasets=compact))
    metadata['status'] = 'complete'
    write_json(args.output_dir / 'run.json', metadata)


if __name__ == '__main__':
    main()
