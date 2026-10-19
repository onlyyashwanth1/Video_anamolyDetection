#!/usr/bin/env python3
"""UCF-Crime evaluation with permanent normal memory and no test-time updates.

Stages: build-memory (all normal videos), evaluate (saved bank), all (both).
Caching/resume and GPU execution verification are deferred.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(dataset="UCF-Crime"):
    import yaml
    from anomaly_detection.evaluation.bank import build_bank, load_bank
    from anomaly_detection.evaluation.dataset import (
        check_disjoint, discover_videos, load_annotations, validate_test_manifest,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build-memory", "evaluate", "all"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--normal-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--annotation-file", "--annotation_file", type=Path)
    parser.add_argument("--memory-bank-path", "--memory_bank_path", type=Path)
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    if dataset == "XD-Violence":
        parser.add_argument("--dataset-root", type=Path,
                            default=Path('/home/nverma/training_free_vad/Datasets/XDViolence'))
        parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = yaml.safe_load(args.config.read_text())
    settings = config["evaluation"]
    settings["dataset"] = dataset
    if dataset == "XD-Violence":
        root = args.dataset_root.expanduser().resolve()
        if args.normal_dir or args.test_dir or args.annotation_file:
            parser.error('For XD use --dataset-root to select the prepared dataset')
        settings.update(normal_dir=str(root / 'Normal_Videos'),
                        test_dir=str(root / 'Testing_Video'),
                        annotation_file=str(root / 'annotations.txt'),
                        annotation_fps='native', llm_review=False,
                        annotation_source_indices='zero-based inclusive',
                        memory_bank_path=('outputs/memory/xd_violence_windows_v2.npz'
                                          if config.get('temporal', {}).get('mode') == 'sliding_windows'
                                          else 'outputs/memory/xd_violence_normal.npz'),
                        output_dir='outputs/xd_violence_evaluation')

    def configured_path(argument, key):
        path = Path(argument or settings[key]).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    normal_dir = configured_path(args.normal_dir, "normal_dir")
    test_dir = configured_path(args.test_dir, "test_dir")
    annotation_path = configured_path(args.annotation_file, "annotation_file")
    bank_path = configured_path(args.memory_bank_path, "memory_bank_path")
    if dataset == "XD-Violence":
        from anomaly_detection.evaluation.xd_dataset import load_dataset
        normal, testing, annotations = load_dataset(root)
    else:
        normal, testing = discover_videos(normal_dir), discover_videos(test_dir)
        annotations = load_annotations(annotation_path)
    check_disjoint(normal, testing)
    validate_test_manifest(testing, annotations)
    if not 0 <= config["inference"]["anomaly_threshold"] <= 1:
        raise ValueError("inference.anomaly_threshold must be in [0, 1]")
    if config["video"]["target_fps"] <= 0:
        raise ValueError("video.target_fps must be positive")
    logging.info("Validated %d normal-memory videos and %d test videos", len(normal), len(testing))
    if dataset == "XD-Violence" and args.preflight_only:
        return

    bank, metadata = None, None
    if bank_path.exists():
        if args.stage == "build-memory":
            raise FileExistsError(f"Permanent memory already exists: {bank_path}. Choose another path to rebuild.")
        bank, metadata = load_bank(bank_path, config, normal)
        logging.info("Using permanent normal memory: %s", bank_path)
    elif args.stage == "evaluate":
        raise FileNotFoundError(f"Build the normal memory first: {bank_path}")

    # Load model dependencies only after dataset/artifact preflight succeeds.
    from anomaly_detection.processing import VideoEventProcessor
    processor = VideoEventProcessor(config, strict=True)
    if bank is None:
        bank, metadata = build_bank(processor, normal, config, bank_path)
    if args.stage == "build-memory":
        return

    from anomaly_detection.evaluation.runner import evaluate
    if args.output_dir:
        output = configured_path(args.output_dir, "output_dir")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = configured_path(None, "output_dir") / stamp
    summary = evaluate(processor, bank, metadata, testing, annotations, config,
                       bank_path, annotation_path, output)
    print(json.dumps(summary["overall"], indent=2))
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
