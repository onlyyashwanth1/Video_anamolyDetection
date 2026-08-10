#!/usr/bin/env python3
"""
CLI entry point. Run from the project root:
    python scripts/run_pipeline.py --video path/to/video.mp4
    python scripts/run_pipeline.py --webcam

Works either after `pip install -e .`, or standalone (the sys.path
fallback below makes the package importable without installing it).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.pipeline import load_config, run_steps_1_to_3  # noqa: E402


def print_tree(tree, indent=0):
    for chunk in tree:
        print("  " * indent + f"- {chunk}")
        if chunk.children:
            print_tree(chunk.children, indent + 1)


def main():
    default_config = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

    parser = argparse.ArgumentParser(description="Run Steps 1-3 of the pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam instead")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    source = 0 if args.webcam else args.video
    if source is None:
        raise SystemExit("Provide --video path/to/file.mp4 or --webcam")

    config = load_config(args.config)
    print(f"Running Steps 1-3 on source: {source}")
    tree = run_steps_1_to_3(source, config)

    print(f"\nFound {len(tree)} top-level events:")
    print_tree(tree)


if __name__ == "__main__":
    main()
