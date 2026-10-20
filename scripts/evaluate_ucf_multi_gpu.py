#!/usr/bin/env python3
"""Resumable single-host evaluation with independent video workers per GPU."""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(dataset="UCF-Crime", argv=None):
    import yaml
    from anomaly_detection.evaluation.bank import load_bank
    from anomaly_detection.evaluation.dataset import (
        check_disjoint, discover_videos, load_annotations, validate_test_manifest,
    )
    from anomaly_detection.evaluation.multi_gpu import (
        cache_lock, finalize, pending_jobs, resolve_devices, run_workers,
    )
    from anomaly_detection.evaluation.video_cache import VideoCache

    if dataset not in ("UCF-Crime", "XD-Violence"):
        raise ValueError(f"Unsupported dataset: {dataset}")
    prefix = "xd_violence" if dataset == "XD-Violence" else "ucf"
    bank_name = "xd_violence" if dataset == "XD-Violence" else "ucf_crime"
    parser = argparse.ArgumentParser(description=f"{dataset}: {__doc__}")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--gpus", required=True, help="Visible GPU indices, e.g. 0,1,2,3,4,5")
    parser.add_argument("--stage", choices=("all", "build-memory", "evaluate"), default="all")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / f"outputs/checkpoints/{prefix}_multi_gpu")
    parser.add_argument("--memory-bank-path", type=Path,
                        default=ROOT / f"outputs/memory/{bank_name}_multi_gpu.npz")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--normal-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--annotation-file", type=Path)
    if dataset == "XD-Violence":
        parser.add_argument("--dataset-root", type=Path,
                            default=Path("/home/nverma/training_free_vad/Datasets/XDViolence"))
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Inspect datasets/checkpoints; no models or GPU use")
    args = parser.parse_args(argv)
    if dataset == "XD-Violence" and (args.normal_dir or args.test_dir or args.annotation_file):
        parser.error("For XD use --dataset-root to select the prepared dataset")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.threads_per_worker < 1:
        parser.error("--threads-per-worker must be positive")
    try:
        devices = resolve_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    except ValueError as error:
        parser.error(str(error))
    config = yaml.safe_load(args.config.read_text())
    settings = config["evaluation"]
    settings["dataset"] = dataset

    def path(value):
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (ROOT / value).resolve()

    if dataset == "XD-Violence":
        from anomaly_detection.evaluation.xd_dataset import load_dataset
        root = path(args.dataset_root)
        settings.update(normal_dir=str(root / "Normal_Videos"),
                        test_dir=str(root / "Testing_Video"),
                        annotation_file=str(root / "annotations.txt"),
                        annotation_fps="native", llm_review=False,
                        annotation_source_indices="zero-based inclusive")
        normal, testing, annotations = load_dataset(root)
        annotation_path = root / "annotations.txt"
    else:
        normal = discover_videos(path(args.normal_dir or settings["normal_dir"]))
        testing = discover_videos(path(args.test_dir or settings["test_dir"]))
        annotation_path = path(args.annotation_file or settings["annotation_file"])
        annotations = load_annotations(annotation_path)
    check_disjoint(normal, testing)
    validate_test_manifest(testing, annotations)
    if not 0 <= config["inference"]["anomaly_threshold"] <= 1 or config["video"]["target_fps"] <= 0:
        parser.error("Invalid threshold or sampling FPS")
    bank_path, cache_dir = path(args.memory_bank_path), path(args.checkpoint_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = path(args.output_dir or f"outputs/{prefix}_multi_gpu_evaluation/{stamp}")
    settings.update(memory_bank_path=str(bank_path), output_dir=str(output))
    if args.stage != "build-memory" and output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; choose a fresh output directory (checkpoints are reused)")
    if bank_path.exists():
        load_bank(bank_path, config, normal)
    elif args.stage == "evaluate":
        parser.error("Build the memory bank first, or use --stage all")

    def prepare():
        videos = {}
        if not bank_path.exists():
            videos.update(normal)
        if args.stage != "build-memory":
            videos.update(testing)
        cache = VideoCache(cache_dir, config)
        jobs, resumed = pending_jobs(videos, cache)
        logging.info("GPUs: %s; validated %d normal / %d test videos", devices, len(normal), len(testing))
        logging.info("%d cached; %d pending; %.2f hours of pending footage",
                     resumed, len(jobs), sum(j["seconds"] for j in jobs) / 3600)
        for job in jobs[:10]:
            logging.info("Queue: %s (%.2f hours)", job["name"], job["seconds"] / 3600)
        return jobs

    if args.dry_run:
        prepare()
        logging.info("Dry run only; GPU availability and real-model throughput have not been tested")
        return
    # Avoid relative model/download paths changing with the launch directory.
    os.chdir(ROOT)
    with cache_lock(cache_dir):
        jobs = prepare()
        run_workers(devices, config, cache_dir, jobs, args.threads_per_worker)
        finalize(devices[0], config, cache_dir, normal, testing, annotations, bank_path,
                 annotation_path, output, args.stage, args.threads_per_worker)
    logging.info("Memory bank: %s", bank_path)
    if args.stage != "build-memory":
        logging.info("Evaluation results: %s", output)


if __name__ == "__main__":
    main()
