#!/usr/bin/env python3
"""Resumable multi-GPU benchmark evaluation for the action and memory_novelty architectures.

    python scripts/evaluate.py --dataset ucf --architecture action_default --gpus 0,1,2
    python scripts/evaluate.py --dataset xd --architecture b_plus_j --gpus 0,1

Stages: GPU workers caption and embed every pending video into the checkpoint
folder (reused across architectures); then one process builds or loads the
normal-memory bank (when the architecture needs one) and scores the test set.
--config replaces --architecture with any config file; the dataset file is
always merged on top.
"""

import argparse
from datetime import datetime, timezone
import logging
import os
import signal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATASETS = {"ucf": "UCF-Crime", "xd": "XD-Violence", "shanghaitech": "ShanghaiTech", "avenue": "Avenue"}


def main(argv=None):
    from anomaly_detection.config import load_config
    from anomaly_detection.evaluation.logging_utils import configure_evaluation_logging
    from anomaly_detection.memory.store import file_digest, load_bank
    from anomaly_detection.evaluation.dataset import (
        check_disjoint, discover_videos, load_annotations, validate_test_manifest,
    )
    from anomaly_detection.evaluation.multi_gpu import (
        cache_lock, finalize, pending_jobs, resolve_devices, run_workers,
    )
    from anomaly_detection.evaluation.video_cache import VideoCache
    from anomaly_detection.scoring import architecture, needs_bank

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--architecture", help="Name of a file in config/architectures/, e.g. b_plus_j")
    model.add_argument("--config", type=Path, help="Any config file (instead of --architecture)")
    parser.add_argument("--gpus", required=True, help="Visible GPU indices, e.g. 0,1,2,3")
    parser.add_argument("--stage", choices=("all", "build-memory", "evaluate"), default="all")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--memory-bank-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--normal-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--annotation-file", type=Path)
    parser.add_argument("--dataset-root", type=Path, help="XD-Violence: prepared dataset folder")
    parser.add_argument("--frozen-reference-memory", action="store_true",
                        help="XD evaluate only: reuse source-domain memory without building/adapting XD memory")
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Inspect datasets/checkpoints; no models or GPU use")
    args = parser.parse_args(argv)
    dataset = DATASETS[args.dataset]
    if args.frozen_reference_memory and (dataset != "XD-Violence" or args.stage != "evaluate"):
        parser.error("--frozen-reference-memory requires --dataset xd and --stage evaluate")
    if dataset == "XD-Violence" and (args.normal_dir or args.test_dir or args.annotation_file):
        parser.error("For XD use --dataset-root to select the prepared dataset")
    if dataset != "XD-Violence" and args.dataset_root:
        parser.error("--dataset-root is only for XD-Violence")
    configure_evaluation_logging()
    if args.threads_per_worker < 1:
        parser.error("--threads-per-worker must be positive")
    try:
        devices = resolve_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    except ValueError as error:
        parser.error(str(error))
    config_path = args.config or ROOT / "config/architectures" / f"{args.architecture}.yaml"
    if not config_path.exists():
        parser.error(f"No such config: {config_path}")
    config = load_config(config_path, [ROOT / "config/datasets" / f"{args.dataset}.yaml"])
    name = architecture(config)
    if name == "observe_detect":
        parser.error("observe_detect scores single videos (scripts/run_video.py); it has no benchmark runner")
    from anomaly_detection.reasoning.ollama_runtime import resolve_model, ollama_servers
    resolve_model(config)
    settings = config["evaluation"]
    settings["dataset"] = dataset
    if args.frozen_reference_memory:
        settings["frozen_reference_memory"] = True
    elif settings.get("frozen_reference_memory"):
        parser.error("Frozen-reference configurations require explicit --frozen-reference-memory")

    def path(value):
        value = Path(value).expanduser()
        return value.resolve() if value.is_absolute() else (ROOT / value).resolve()

    if dataset == "XD-Violence":
        from anomaly_detection.evaluation.xd_dataset import load_dataset
        root = path(args.dataset_root or settings["dataset_root"])
        settings.update(normal_dir=str(root / "Normal_Videos"), test_dir=str(root / "Testing_Video"),
                        annotation_file=str(root / "annotations.txt"), annotation_fps="native",
                        llm_review=False, annotation_source_indices="zero-based inclusive")
        normal, testing, annotations = load_dataset(root)
        annotation_path = root / "annotations.txt"
    else:
        normal = discover_videos(path(args.normal_dir or settings["normal_dir"]))
        testing = discover_videos(path(args.test_dir or settings["test_dir"]))
        annotation_path = path(args.annotation_file or settings["annotation_file"])
        annotations = load_annotations(annotation_path)
    check_disjoint(normal, testing)
    validate_test_manifest(testing, annotations)
    if config["video"]["target_fps"] <= 0:
        parser.error("Invalid sampling FPS")
    bank_needed = needs_bank(config) or args.stage == "build-memory"
    bank_path = path(args.memory_bank_path or settings["memory_bank_path"])
    cache_dir = path(args.checkpoint_dir or settings["checkpoint_dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = path(args.output_dir or Path("outputs/evaluations") / args.dataset / config_path.stem / stamp)
    settings.update(memory_bank_path=str(bank_path), output_dir=str(output), checkpoint_dir=str(cache_dir))
    if args.stage != "build-memory" and output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; choose a fresh output directory (checkpoints are reused)")
    if bank_needed and bank_path.exists():
        if args.frozen_reference_memory and settings.get("reference_memory_sha256"):
            if file_digest(bank_path) != settings["reference_memory_sha256"]:
                parser.error("Frozen source memory hash differs from the transfer configuration")
        _, reference_metadata = load_bank(bank_path, config, None if args.frozen_reference_memory else normal)
        if args.frozen_reference_memory:
            reference = {v["name"]: Path(v["path"]) for v in reference_metadata["normal_videos"]}
            check_disjoint(reference, testing)
            logging.info("Frozen source memory: %s; no XD normal videos used to rebuild/calibrate memory", bank_path)
    elif bank_needed and args.stage == "evaluate":
        parser.error("Build the memory bank first, or use --stage all")
    logging.info("Architecture: %s (%s)%s", name, config_path,
                 f"; bank {bank_path}" if bank_needed else "; no memory bank needed")

    def prepare():
        videos = {}
        if bank_needed and not bank_path.exists():
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

    def terminate(signum, frame):  # let managed servers and workers clean up on screen shutdown
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    with cache_lock(cache_dir):
        jobs = prepare()
        with ollama_servers(devices, config, ROOT / f"outputs/ollama_logs/{args.dataset}/{stamp}"):
            run_workers(devices, config, cache_dir, jobs, args.threads_per_worker)
            finalize(devices[0], config, cache_dir, normal, testing, annotations, bank_path,
                     annotation_path, output, args.stage, args.threads_per_worker)
    if bank_needed:
        logging.info("Memory bank: %s", bank_path)
    if args.stage != "build-memory":
        logging.info("Evaluation results: %s", output)


if __name__ == "__main__":
    main()
