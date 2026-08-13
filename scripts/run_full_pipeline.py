#!/usr/bin/env python3
"""
Runs the FULL pipeline (Steps 1-9) on a video or webcam:
    video -> fingerprints -> event tree -> "Normal" notebook -> anomaly scoring

Usage:
    python scripts/run_full_pipeline.py --video path/to/video.mp4
    python scripts/run_full_pipeline.py --webcam

Note: needs a video with at least 2 top-level events (see
segmentation.coarse_threshold in config.yaml if your video only produces 1).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.pipeline import load_config, run_full_pipeline  # noqa: E402


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


def main():
    default_config = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

    parser = argparse.ArgumentParser(description="Run the full Steps 1-9 pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam instead")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    source = 0 if args.webcam else args.video
    if source is None:
        raise SystemExit("Provide --video path/to/file.mp4 or --webcam")

    config = load_config(args.config)
    print(f"Running full pipeline (Steps 1-9) on source: {source}")

    notebook, memory_bank, results = run_full_pipeline(source, config)

    print_notebook(notebook)
    print_results(results)

    print(f"\n=== Final Memory Bank ({len(memory_bank)} entries) ===")
    print(memory_bank)


if __name__ == "__main__":
    main()
