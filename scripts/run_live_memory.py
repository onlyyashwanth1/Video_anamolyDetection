#!/usr/bin/env python3
"""Observe the first 40% of a video, then flag anomalies in the rest; write an SRT.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_live_memory.py --video path/to/video.mp4
    ... --observe-fraction 0.3          # use 30% for memory instead
    ... --observe-seconds 60            # or a fixed observation time
    ... --no-judge                      # memory distances only (no extra Gemma ratings)
    ... --render                        # also write a captioned .mp4 (OpenCV, no audio)

Outputs go to outputs/live/<video>_<UTC time>/ : <video>.srt, report.json, summary.txt.
Play with:  vlc video.mp4 --sub-file outputs/live/.../video.srt
       or:  mpv video.mp4 --sub-file=outputs/live/.../video.srt
"""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection.pipeline import load_config  # noqa: E402
from anomaly_detection.live.runner import run_live  # noqa: E402
from anomaly_detection.utils.video_source import cleanup_video_source, resolve_video_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Video file or public Google Drive link")
    parser.add_argument("--config", default=str(ROOT / "config/live_memory40.yaml"))
    parser.add_argument("--output-dir", help="Folder for results (must not exist)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--observe-fraction", type=float, help="Share of the video used as memory (default 0.4)")
    group.add_argument("--observe-seconds", type=float, help="Fixed observation time instead of a fraction")
    parser.add_argument("--threshold", type=float, help="Alert when the 0-1 anomaly score is above this (default 0.5)")
    parser.add_argument("--no-judge", action="store_true", help="Disable the memory-aware Gemma judge")
    parser.add_argument("--explain", action="store_true", help="Add a Gemma 'Why:' line to anomaly cues")
    parser.add_argument("--plain-srt", action="store_true", help="No <font color> tags in the SRT")
    parser.add_argument("--render", action="store_true", help="Also write a captioned copy of the video")
    parser.add_argument("--port", type=int, help="First Ollama port (default from config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    live = config["live"]
    if args.observe_fraction is not None:
        live["observe_fraction"], live["observe_seconds"] = args.observe_fraction, None
    elif args.observe_seconds is not None:
        live["observe_fraction"], live["observe_seconds"] = None, args.observe_seconds
    else:
        live.setdefault("observe_seconds", None)
    if args.threshold is not None:
        live["threshold"] = args.threshold
    if args.no_judge:
        live["judge"]["enabled"] = False
        live["judge_weight"] = 0.0
    if args.explain:
        live["explain_anomalies"] = True
    if args.plain_srt:
        live["srt"]["colors"] = False
    if args.port:
        config["reasoning"]["base_port"] = args.port

    source, temporary = resolve_video_source(args.video)
    try:
        stem = Path(source).stem
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(args.output_dir or ROOT / "outputs/live" / f"{stem}_{stamp}")
        run_live(source, config, output, log=lambda line: print(line, flush=True), render_video=args.render)
        print(f"\nPlay:  vlc \"{os.path.abspath(source) if not temporary else args.video}\" "
              f"--sub-file \"{output / (stem + '.srt')}\"")
    finally:
        if temporary:
            cleanup_video_source(temporary)


if __name__ == "__main__":
    main()
