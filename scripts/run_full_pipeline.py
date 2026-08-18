#!/usr/bin/env python3
"""
Runs the FULL pipeline (Steps 1-10) on a local video, public Google Drive
video link, or webcam.

Usage:
    python scripts/run_full_pipeline.py --video path/to/video.mp4
    python scripts/run_full_pipeline.py --video "https://drive.google.com/file/d/FILE_ID/view"
    python scripts/run_full_pipeline.py --webcam

Public/shared Google Drive links are downloaded to a temporary local file,
processed by the existing pipeline, and deleted automatically afterward.
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.pipeline import load_config, run_full_pipeline  # noqa: E402
from anomaly_detection.utils.video_source import (  # noqa: E402
    cleanup_video_source,
    resolve_video_source,
)


def print_notebook(notebook):
    print(f"\n=== 'Normal' Notebook (from first {len(notebook)} distinct patterns seen) ===")
    if not notebook:
        print("  (empty - nothing was captioned during familiarisation)")
    for line in notebook:
        print(f"  - {line}")


def print_results(results):
    print(f"\n=== Live Event Results ({len(results)} events tested) ===")
    for r in results:
        flag = "ANOMALY" if r.is_anomaly else "normal "
        print(f"[{flag}] t={r.event.start_time:.1f}-{r.event.end_time:.1f}s "
              f"  score={r.anomaly_score:.3f}  seen as: \"{r.event.description}\"")
        print(f"           closest match in notebook: \"{r.closest_match}\"")
        if r.explanation:
            print(f"           Explanation: {r.explanation}")


def main():
    default_config = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

    parser = argparse.ArgumentParser(description="Run the full Steps 1-10 pipeline")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file or a public Google Drive sharing URL",
    )
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam instead")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    source = 0 if args.webcam else args.video
    if source is None:
        raise SystemExit("Provide --video path/to/file.mp4, --video <Google Drive URL>, or --webcam")

    config = load_config(args.config)

    resolved_source, temporary_path = resolve_video_source(source)
    if temporary_path:
        print("Google Drive link detected; downloading video to temporary storage...")

    try:
        print(f"Running full pipeline (Steps 1-10) on source: {source}")
        notebook, memory_bank, results = run_full_pipeline(resolved_source, config)

        print_notebook(notebook)
        print_results(results)

        print(f"\n=== Final Memory Bank ({len(memory_bank)} entries) ===")
        print(memory_bank)
    finally:
        if temporary_path:
            cleanup_video_source(temporary_path)
            print("Temporary Google Drive video removed.")


if __name__ == "__main__":
    main()
