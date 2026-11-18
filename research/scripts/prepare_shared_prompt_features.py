#!/usr/bin/env python3
"""Offline shared CLIP prompts for two benchmarks, with exact window provenance.

No labels are used to construct prompts or encode features. All text is encoded
using the locally cached, frozen CLIP text tower. The same prompt definitions
and text weights are applied to both datasets. Files are resumable.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import multiprocessing

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anomaly_detection.evaluation.bank import model_spec

DEFAULT_OUTPUT = ROOT / 'outputs/results/shared_prompt_features_20261026'
RUNS = dict(ucf=ROOT / 'outputs/ucf_expanded95_evaluation/20261022T192245Z',
            xd=ROOT / 'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z')


def prompt_definitions():
    import yaml
    expanded = yaml.safe_load((ROOT / 'config/ucf_action_prompts_expanded.yaml').read_text())
    normal = expanded['normal_prompts']
    anomaly = expanded['anomalous_prompts']
    # Both benchmark vocabularies are included in this single shared semantic set.
    semantic_anomaly = [
        'a person abusing or physically hurting another person',
        'a person being arrested by police officers',
        'a person setting a building or vehicle on fire',
        'people violently assaulting another person',
        'a person breaking into a building',
        'a large explosion with flames and smoke',
        'people fighting with punches and kicks',
        'a road accident with vehicles colliding',
        'a person robbing another person with threats or violence',
        'a person shooting a gun',
        'a person stealing merchandise from a shop',
        'a person stealing belongings or a vehicle',
        'a person intentionally damaging property',
        'people rioting and violently damaging their surroundings',
        'people being shot by gunfire',
        'a vehicle colliding with another vehicle or a pedestrian',
    ]
    semantic_normal = [
        'people walking or standing peacefully',
        'people talking or sitting together peacefully',
        'people exercising or playing sports safely',
        'customers shopping and paying for goods',
        'workers carrying objects or doing ordinary work',
        'vehicles driving safely along a road',
        'vehicles parked beside a road or building',
        'pedestrians crossing a road safely',
        'a peaceful indoor or outdoor scene',
        'a quiet scene with no dangerous activity',
        'people entering or leaving a building normally',
        'people opening doors or handling their own belongings',
    ]
    scene_normal = ['normal everyday activity', 'a safe and peaceful scene',
                    'people behaving normally', 'traffic moving safely',
                    'an ordinary scene without violence or accidents',
                    'people shopping, walking, waiting, or working normally']
    scene_anomaly = ['a violent fight or physical assault', 'a shooting or armed robbery',
                     'a traffic accident or vehicle crash', 'a fire or explosion',
                     'a burglary, theft, or shoplifting', 'vandalism or destruction of property',
                     'a riot or violent crowd', 'an arrest or police chase',
                     'abuse or physical violence against a person']
    variants = {
        'expanded26': dict(normal=normal, anomalous=anomaly),
        'original6': dict(normal=normal[:6], anomalous=anomaly[:6]),
        'semantic_shared16': dict(normal=semantic_normal, anomalous=semantic_anomaly),
        'scene_events': dict(normal=scene_normal, anomalous=scene_anomaly),
        'generic_safety': dict(normal=['a normal scene', 'a safe scene', 'a peaceful scene',
                                       'ordinary activity'],
                               anomalous=['an abnormal scene', 'a dangerous scene',
                                          'a violent scene', 'a crime or accident']),
    }
    for base in ('expanded26', 'semantic_shared16', 'scene_events'):
        variants[base + '_video'] = {
            k: ['a video of ' + phrase for phrase in v] for k, v in variants[base].items()}
    return variants


def atomic_npz(path, **arrays):
    temp = path.with_suffix('.writing.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


_WORKER_TEXT_MODEL = None
_WORKER_TOKENIZER = None


def encode_caption_shard(task):
    """Separate CPU workers avoid one text tower's thread scaling ceiling."""
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    global _WORKER_TEXT_MODEL, _WORKER_TOKENIZER
    start, texts, output_name, threads, batch_size = task
    output_name = Path(output_name)
    if output_name.exists():
        saved = np.load(output_name, mmap_mode='r')
        if saved.shape != (len(texts), 768):
            raise ValueError('Invalid completed text shard')
        return start, start + len(texts), str(output_name)
    torch.set_num_threads(threads)
    if _WORKER_TEXT_MODEL is None:
        _WORKER_TOKENIZER = CLIPTokenizerFast.from_pretrained(
            'openai/clip-vit-large-patch14', local_files_only=True)
        _WORKER_TEXT_MODEL = CLIPTextModelWithProjection.from_pretrained(
            'openai/clip-vit-large-patch14', local_files_only=True).eval()
    tokenized = _WORKER_TOKENIZER(texts, padding=False)
    lengths = np.asarray([len(x) for x in tokenized['input_ids']])
    if lengths.max() > 77:
        raise ValueError('Text exceeds baseline CLIP context')
    order = np.argsort(lengths, kind='stable')
    vectors = np.empty((len(texts), 768), dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            rows = order[offset:offset + batch_size]
            inputs = _WORKER_TOKENIZER([texts[i] for i in rows], padding=True,
                                       return_tensors='pt')
            features = _WORKER_TEXT_MODEL(**inputs).text_embeds
            features = features / features.norm(dim=-1, keepdim=True)
            vectors[rows] = features.cpu().numpy()
    temporary = output_name.with_suffix('.writing.npy')
    np.save(temporary, vectors)
    os.replace(temporary, output_name)
    return start, start + len(texts), str(output_name)


def parallel_caption_texts(texts, output, batch_size, workers, threads,
                          validation_count, on_progress):
    path = output / 'caption_embeddings.npy'
    progress = output / 'caption_progress.json'
    digest = hashlib.sha256('\n'.join(texts).encode()).hexdigest()
    start_row = 0
    if path.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state['texts_sha256'] != digest or state['total'] != len(texts):
            raise ValueError('Caption cache identity differs')
        start_row = state['completed']
        vectors = np.lib.format.open_memmap(path, mode='r+')
    else:
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                           shape=(len(texts), 768))
    shard_dir = output / 'caption_shards'
    shard_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    if on_progress:
        on_progress(vectors, start_row)
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        # Complete validation first, then all remaining captions. Every shard is
        # length-bucketed internally, while final embedding rows remain fixed.
        completed = start_row
        for phase_stop in (validation_count, len(texts)):
            if completed >= phase_stop:
                continue
            futures = []
            for first in range(completed, phase_stop, 4096):
                last = min(first + 4096, phase_stop)
                shard_path = shard_dir / f'{digest[:12]}_{first}_{last}.npy'
                futures.append(pool.submit(encode_caption_shard,
                    (first, texts[first:last], str(shard_path), threads, batch_size)))
            finished = {}
            for future in as_completed(futures):
                first, last, shard_path = future.result()
                vectors[first:last] = np.load(shard_path, mmap_mode='r')
                finished[first] = last
                while completed in finished:
                    completed = finished.pop(completed)
                vectors.flush()
                progress.write_text(json.dumps(dict(total=len(texts), completed=completed,
                                                     texts_sha256=digest), indent=2) + '\n')
                print(f'caption workers: contiguous {completed}/{len(texts)}; '
                      f'shard {first}:{last} ready; {time.monotonic()-started:.1f}s', flush=True)
                if on_progress:
                    on_progress(vectors, completed)
    return np.asarray(vectors)


def load_windows(key, folder):
    run = json.loads((folder / 'run.json').read_text())
    names = sorted(json.loads((folder / 'summary.json').read_text())['per_video'])
    cache_dir = Path(run['config']['evaluation']['checkpoint_dir'])
    cache_dir = cache_dir if cache_dir.is_absolute() else ROOT / cache_dir
    spec = model_spec(run['config'])
    images, captions, starts, stops, video_indices, offsets = [], [], [], [], [], [0]
    original_image, original_text = [], []
    # Manifests carry the exact source identity used to name cache NPZ files.
    sources = {x['name']: x for x in run['test_videos']}
    for i, name in enumerate(names):
        source = sources[name]
        identity = dict(version=1, model=spec, path=source['path'],
                        bytes=source['bytes'], mtime_ns=source['mtime_ns'])
        hashed = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = cache_dir / (hashed + '.npz')
        if not path.exists():
            raise FileNotFoundError(f'Cache identity mismatch for {name}: {path}')
        data = json.loads((folder / 'predictions' / (name + '.json')).read_text())
        with np.load(path, allow_pickle=False) as z:
            vectors = z['embeddings']
            metadata = json.loads(str(z['metadata'].item()))
        events = data['events']
        if len(vectors) != len(events) or vectors.shape[1] != 768:
            raise ValueError('Cache/prediction rows differ')
        if metadata['identity'] != identity:
            raise ValueError('Cache source identity differs')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Cache embeddings not normalized')
        for e, cached in zip(events, metadata['events']):
            if (e['start_frame'], e['stop_frame'], e['caption']) != (
                    cached['start_frame'], cached['stop_frame'], cached['description']):
                raise ValueError('Cache/prediction window identity differs')
        images.append(vectors)
        captions.extend(e['caption'] for e in events)
        starts.extend(e['start_frame'] for e in events)
        stops.extend(e['stop_frame'] for e in events)
        video_indices.extend([i] * len(events))
        offsets.append(offsets[-1] + len(events))
        original_image.extend(e['components']['action_image_margin'] for e in events)
        original_text.extend(e['components']['action_text_margin'] for e in events)
        if (i + 1) % 100 == 0:
            print(f'{key}: loaded {i+1}/{len(names)} videos', flush=True)
    return dict(names=np.asarray(names), offsets=np.asarray(offsets),
                start_frame=np.asarray(starts), stop_frame=np.asarray(stops),
                video_index=np.asarray(video_indices), images=np.concatenate(images),
                captions=np.asarray(captions),
                original_image_margin=np.asarray(original_image),
                original_text_margin=np.asarray(original_text))


def encode_texts(texts, output, prefix, batch_size=128, on_progress=None):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
    torch.set_num_threads(4)
    name = 'openai/clip-vit-large-patch14'
    tokenizer = CLIPTokenizerFast.from_pretrained(name, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(name, local_files_only=True).eval()
    path = output / (prefix + '_embeddings.npy')
    progress = output / (prefix + '_progress.json')
    digest = hashlib.sha256('\n'.join(texts).encode()).hexdigest()
    start_row = 0
    if path.exists() and progress.exists():
        state = json.loads(progress.read_text())
        if state['texts_sha256'] != digest or state['total'] != len(texts):
            raise ValueError('Text cache identity differs')
        start_row = state['completed']
        vectors = np.lib.format.open_memmap(path, mode='r+')
    else:
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                           shape=(len(texts), 768))
    started = time.monotonic()
    if on_progress is not None:
        on_progress(vectors, start_row)
    with torch.inference_mode():
        for start in range(start_row, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, return_tensors='pt')
            if inputs['input_ids'].shape[1] > 77:
                raise ValueError('Text exceeds the baseline CLIP context; refusing silent truncation')
            features = model(**inputs).text_embeds
            features = features / features.norm(dim=-1, keepdim=True)
            stop = start + len(batch)
            vectors[start:stop] = features.cpu().numpy()
            if start == start_row or stop == len(texts) or stop // 2048 != start // 2048:
                vectors.flush()
                progress.write_text(json.dumps(dict(total=len(texts), completed=stop,
                                                     texts_sha256=digest), indent=2) + '\n')
                print(f'{prefix}: {stop}/{len(texts)} texts in {time.monotonic()-started:.1f}s', flush=True)
                if on_progress is not None:
                    on_progress(vectors, stop)
    return np.asarray(vectors)


def margins(similarities, variants, prompt_ids):
    result = []
    for definition in variants.values():
        pos = [prompt_ids[p] for p in definition['anomalous']]
        neg = [prompt_ids[p] for p in definition['normal']]
        result.append(similarities[:, pos].max(axis=1) - similarities[:, neg].max(axis=1))
    return np.stack(result, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--image-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = prompt_definitions()
    prompts = list(dict.fromkeys(p for definition in variants.values()
                                 for group in definition.values() for p in group))
    prompt_ids = {p: i for i, p in enumerate(prompts)}
    provenance = dict(model='openai/clip-vit-large-patch14', variants=variants,
                      variant_names=list(variants), prompts=prompts,
                      prompt_ids=prompt_ids, threads=4, label_use='none during feature creation',
                      prompt_definitions_fixed_before_metric_inspection=True,
                      source_runs={k: str(p) for k, p in RUNS.items()})
    (args.output / 'prompt_features.json').write_text(json.dumps(provenance, indent=2) + '\n')
    prompt_vectors = encode_texts(prompts, args.output, 'prompt', args.batch_size)
    all_windows = {}
    for key, folder in RUNS.items():
        cached_windows = args.output / (key + '_windows.npz')
        if cached_windows.exists():
            with np.load(cached_windows, allow_pickle=False) as z:
                windows = {k: z[k] for k in z.files}
        else:
            windows = load_windows(key, folder)
            atomic_npz(cached_windows, **windows)
        similarities = windows['images'] @ prompt_vectors.T
        image_margins = margins(similarities, variants, prompt_ids)
        err = float(np.max(np.abs(image_margins[:, 0] - windows['original_image_margin'])))
        if err > 2e-6:
            raise ValueError(f'{key}: baseline image margins differ by {err}')
        index = {k: windows[k] for k in ('names', 'offsets', 'start_frame', 'stop_frame', 'video_index')}
        atomic_npz(args.output / (key + '_image_prompt_features.npz'), **index,
                   variant_names=np.asarray(list(variants)), image_margins=image_margins,
                   image_similarities=similarities,
                   action_image_margin=image_margins, original_text_margin=windows['original_text_margin'])
        print(f'{key}: image prompts ready; {len(image_margins)} windows, baseline error {err:.3g}', flush=True)
        all_windows[key] = windows
    if args.image_only:
        return
    # Encode validation captions first to let joint search start promptly. Text
    # duplicates across datasets are shared, and all remaining rows follow once.
    split_file = ROOT / 'outputs/results/score_ablations_20261026/ablations.json'
    splits = json.loads(split_file.read_text())
    validation = {('ucf' if x['dataset'] == 'UCF-Crime' else 'xd'):
                  set(x['diagnostic_validation_videos']) for x in splits}
    ordered_captions = []
    validation_caption_count = 0
    for validation_only in (True, False):
        for key, windows in all_windows.items():
            for vi, name in enumerate(windows['names']):
                if (name in validation[key]) != validation_only:
                    continue
                a, b = windows['offsets'][vi:vi+2]
                ordered_captions.extend(windows['captions'][a:b].tolist())
        if validation_only:
            validation_caption_count = len(set(ordered_captions))
    captions = list(dict.fromkeys(ordered_captions))
    (args.output / 'caption_texts.json').write_text(json.dumps(captions, ensure_ascii=False) + '\n')
    caption_lookup = {c: i for i, c in enumerate(captions)}
    exported_validation = False

    def export_validation(vectors, completed):
        nonlocal exported_validation
        if exported_validation or completed < validation_caption_count:
            return
        for key, windows in all_windows.items():
            rows = np.asarray([caption_lookup[c] for c in windows['captions']])
            available = rows < validation_caption_count
            text_margins = np.full((len(rows), len(variants)), np.nan, dtype=np.float32)
            text_similarities = np.full((len(rows), len(prompt_vectors)), np.nan, dtype=np.float32)
            text_similarities[available] = vectors[rows[available]] @ prompt_vectors.T
            text_margins[available] = margins(text_similarities[available], variants, prompt_ids)
            image_similarities = windows['images'] @ prompt_vectors.T
            image_margins = margins(image_similarities, variants, prompt_ids)
            index = {k: windows[k] for k in ('names', 'offsets', 'start_frame', 'stop_frame', 'video_index')}
            atomic_npz(args.output / (key + '_validation_prompt_features.npz'), **index,
                       available=available, variant_names=np.asarray(list(variants)),
                       image_margins=image_margins, text_margins=text_margins,
                       image_similarities=image_similarities, text_similarities=text_similarities,
                       action_image_margin=image_margins, action_text_margin=text_margins)
        exported_validation = True
        print(f'Validation prompt features ready; {validation_caption_count} unique captions', flush=True)

    if args.workers > 1:
        caption_vectors = parallel_caption_texts(captions, args.output, args.batch_size,
                                                args.workers, args.threads,
                                                validation_caption_count, export_validation)
    else:
        caption_vectors = encode_texts(captions, args.output, 'caption', args.batch_size,
                                       on_progress=export_validation)
    for key, windows in all_windows.items():
        rows = np.asarray([caption_lookup[c] for c in windows['captions']])
        text_similarities = caption_vectors[rows] @ prompt_vectors.T
        text_margins = margins(text_similarities, variants, prompt_ids)
        image_similarities = windows['images'] @ prompt_vectors.T
        image_margins = margins(image_similarities, variants, prompt_ids)
        err = float(np.max(np.abs(text_margins[:, 0] - windows['original_text_margin'])))
        if err > 3e-6:
            raise ValueError(f'{key}: baseline text margins differ by {err}')
        index = {k: windows[k] for k in ('names', 'offsets', 'start_frame', 'stop_frame', 'video_index')}
        atomic_npz(args.output / (key + '_prompt_features.npz'), **index,
                   variant_names=np.asarray(list(variants)), image_margins=image_margins,
                   text_margins=text_margins, action_image_margin=image_margins,
                   action_text_margin=text_margins, image_similarities=image_similarities,
                   text_similarities=text_similarities, caption_embedding_row=rows)
        print(f'{key}: all prompt features ready; baseline text error {err:.3g}', flush=True)


if __name__ == '__main__':
    main()
