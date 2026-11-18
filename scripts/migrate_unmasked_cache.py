#!/usr/bin/env python3
"""Build unmasked video checkpoints from the old (masked) ones without re-running Gemma.

Gemma always captioned raw frames, so old checkpoints' window boundaries and
captions are still valid. Only the CLIP window embeddings change. This script
pairs each old checkpoint with the unmasked embeddings already extracted to
outputs/features/unmasked_clip_v1 (same 1 fps sampling, multi-scale CLIP and
4 s / 1 s windows; window starts and stops are checked) and writes a checkpoint
in the new format to the dataset's checkpoint_dir.

Videos without saved unmasked embeddings (e.g. the 20% held-out calibration
normals) are listed; --compute-missing encodes them with CLIP on --gpu (no Gemma).

    python scripts/migrate_unmasked_cache.py --dataset ucf --dry-run
    python scripts/migrate_unmasked_cache.py --dataset ucf --compute-missing --gpu 0
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OLD_CHECKPOINTS = {"ucf": "outputs/checkpoints/ucf_ollama",
                   "xd": "outputs/checkpoints/xd_ollama_fresh_20261022T192245Z",
                   "shanghaitech": "outputs/checkpoints/shanghaitech_ollama",
                   "avenue": "outputs/checkpoints/avenue_ollama"}
UNMASKED_NAMES = {"ucf": "ucf", "xd": "xd", "shanghaitech": "sht", "avenue": "avenue"}
SHARED_SPEC_KEYS = ("encoder", "video_target_fps", "caption_model", "captioning", "strict_models")


def compatible(old_spec, new_spec):
    """Old features differ only by masking; everything that shaped the captions must match."""
    problems = [k for k in SHARED_SPEC_KEYS if old_spec.get(k) != new_spec.get(k)]
    old_temporal = {k: old_spec.get("temporal", {}).get(k) for k in new_spec["temporal"]}
    if old_temporal != new_spec["temporal"]:
        problems.append("temporal")
    return problems


def unmasked_windows(folder, name):
    for split in ("test", "bank"):
        path = folder / split / f"{name}.npz"
        if path.exists():
            with np.load(path) as stored:
                return stored["embeddings"].astype(np.float32), stored["start_frame"], stored["stop_frame"]
    return None


def main(argv=None):
    from anomaly_detection.config import load_config
    from anomaly_detection.datatypes import EventChunk
    from anomaly_detection.evaluation.video_cache import VideoCache
    from anomaly_detection.memory.store import feature_spec
    from anomaly_detection.reasoning.ollama_runtime import resolve_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(OLD_CHECKPOINTS))
    parser.add_argument("--old-checkpoint-dir", type=Path)
    parser.add_argument("--unmasked-dir", type=Path, default=ROOT / "outputs/features/unmasked_clip_v1")
    parser.add_argument("--compute-missing", action="store_true", help="Encode missing videos with CLIP (GPU)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written; write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(ROOT / "config/base.yaml", [ROOT / "config/datasets" / f"{args.dataset}.yaml"])
    resolve_model(config)
    new_spec = feature_spec(config)
    old_dir = args.old_checkpoint_dir or ROOT / OLD_CHECKPOINTS[args.dataset]
    new_dir = ROOT / config["evaluation"]["checkpoint_dir"]
    unmasked = args.unmasked_dir / UNMASKED_NAMES[args.dataset]
    cache = VideoCache(new_dir, config)
    processor = None
    counts = {"written": 0, "already_done": 0, "computed": 0}
    skipped = {"incompatible": [], "source_changed": [], "no_unmasked_features": [], "window_mismatch": []}
    files = sorted(old_dir.glob("*.npz"))
    logging.info("%d old checkpoints in %s -> %s", len(files), old_dir, new_dir)
    for number, old_file in enumerate(files, 1):
        with np.load(old_file, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
        identity = metadata["identity"]
        source = Path(identity["path"])
        name = source.stem
        problems = compatible(identity["model"], new_spec)
        if problems:
            skipped["incompatible"].append(f"{name}: {problems}")
            continue
        if not source.exists() or {k: identity[k] for k in ("bytes", "mtime_ns")} != \
                {k: cache.identity(source)[k] for k in ("bytes", "mtime_ns")}:
            skipped["source_changed"].append(name)
            continue
        if cache.load(source) is not None:
            counts["already_done"] += 1
            continue
        records = metadata["events"]
        starts = np.asarray([r["start_frame"] for r in records])
        stops = np.asarray([r["stop_frame"] for r in records])
        found = unmasked_windows(unmasked, name)
        if found is None:
            if not args.compute_missing:
                skipped["no_unmasked_features"].append(name)
                continue
            if args.dry_run:
                counts["computed"] += 1
                continue
            if processor is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                from anomaly_detection.perception.processor import VideoEventProcessor
                processor = VideoEventProcessor(config, strict=True)
            video = processor.process(str(source), caption=False)
            found = (np.stack([e.average_embedding for e in video.events]).astype(np.float32),
                     np.asarray([e.start_frame for e in video.events]),
                     np.asarray([e.stop_frame for e in video.events]))
            counts["computed"] += 1
        embeddings, new_starts, new_stops = found
        if not (np.array_equal(new_starts, starts) and np.array_equal(new_stops, stops)):
            skipped["window_mismatch"].append(name)
            continue
        if not args.dry_run:
            events = [EventChunk(**r, average_embedding=v) for r, v in zip(records, embeddings)]
            cache.save(source, SimpleNamespace(events=events, frame_count=metadata["frame_count"],
                                               fps=metadata["fps"]), expected_identity=cache.identity(source))
        counts["written"] += 1
        if number % 200 == 0:
            logging.info("%d/%d checked", number, len(files))
    report = {"dataset": args.dataset, "dry_run": args.dry_run, "old_checkpoints": len(files), **counts,
              **{k: len(v) for k, v in skipped.items()},
              "examples": {k: v[:5] for k, v in skipped.items() if v}}
    print(json.dumps(report, indent=2))
    if skipped["no_unmasked_features"]:
        print("Videos without saved unmasked features need --compute-missing (CLIP only, no Gemma).")


if __name__ == "__main__":
    main()
