#!/usr/bin/env python3
"""
Score one video (local file, public Google Drive link, or webcam) with any architecture.

Usage:
    python scripts/run_video.py --video path/to/video.mp4                      # observe_detect
    python scripts/run_video.py --video clip.mp4 --architecture action_default --dataset ucf
    python scripts/run_video.py --video clip.mp4 --architecture b_plus_j --dataset ucf
    python scripts/run_video.py --webcam

action and memory_novelty need a saved normal-memory bank, taken from the
--dataset file (evaluation.memory_bank_path). Google Drive downloads are deleted
afterwards. Results are printed, saved as text, and written as SRT subtitles.
"""

import argparse
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.config import load_config  # noqa: E402
from anomaly_detection.pipeline import run_full_pipeline  # noqa: E402
from anomaly_detection.perception.video import (  # noqa: E402
    cleanup_video_source,
    resolve_video_source,
)


def print_notebook(notebook):
    print(f"\n=== Normal memory captions ({len(notebook)}) ===")
    if not notebook:
        print("  (none: this architecture uses no caption memory)")
    for line in notebook[:200]:
        print(f"  - {line}")
    if len(notebook) > 200:
        print(f"  ... {len(notebook) - 200} more")


def print_results(results):
    print(f"\n=== Live Event Results ({len(results)} events tested) ===")
    for r in results:
        flag = "ANOMALY" if r.is_anomaly else "normal "
        print(f"[{flag}] t={r.event.start_time:.1f}-{r.event.end_time:.1f}s "
              f"  score={r.anomaly_score:.3f}  seen as: \"{r.event.description}\"")
        if r.closest_match:
            print(f"           closest normal memory: \"{r.closest_match}\"")
        if r.explanation:
            print(f"           Explanation: {r.explanation}")


def build_output(notebook, memory_bank, results, source):
    """Build the complete human-readable pipeline report."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        print("=" * 72)
        print("VIDEO ANOMALY DETECTION - PIPELINE RESULTS")
        print("=" * 72)
        print(f"Source: {source}")
        print_notebook(notebook)
        print_results(results)
        print(f"\n=== Memory: {len(memory_bank)} caption entries ===")
        print("\n" + "=" * 72)
        print("END OF REPORT")
        print("=" * 72)
    return buffer.getvalue()


def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def export_to_srt(results, output_srt_path):
    """Generate an SRT subtitle file from the anomaly results."""
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            start_ts = format_timestamp(r.event.start_time)
            end_ts = format_timestamp(r.event.end_time)
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            
            flag = "ANOMALY" if r.is_anomaly else "NORMAL"
            f.write(f"[{flag}] Score: {r.anomaly_score:.3f}\n")
            f.write(f"Caption: {r.event.description}\n\n")


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    default_output = os.path.join(os.path.dirname(__file__), "..", "outputs", "pipeline_results.txt")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file or a public Google Drive sharing URL",
    )
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam instead")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--architecture", default="observe_detect",
                       help="Name of a file in config/architectures/ (default: observe_detect)")
    model.add_argument("--config", type=str, help="Any config file (instead of --architecture)")
    parser.add_argument("--dataset", choices=("ucf", "xd", "shanghaitech", "avenue"),
                        help="Dataset whose saved normal-memory bank to use (action, memory_novelty)")
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path for the text output report (default: outputs/pipeline_results.txt)",
    )
    parser.add_argument("--srt", action="store_true", default=True, help="Generate an SRT subtitle file alongside the text report")
    parser.add_argument("--burn", action="store_true", help="Use FFmpeg to burn the SRT subtitles directly into a new output video")
    args = parser.parse_args()

    source = 0 if args.webcam else args.video
    if source is None:
        raise SystemExit("Provide --video path/to/file.mp4, --video <Google Drive URL>, or --webcam")

    config_path = args.config or os.path.join(root, "config", "architectures", f"{args.architecture}.yaml")
    overlays = [os.path.join(root, "config", "datasets", f"{args.dataset}.yaml")] if args.dataset else []
    config = load_config(config_path, overlays)

    resolved_source, temporary_path = resolve_video_source(source)
    if temporary_path:
        print("Google Drive link detected; downloading video to temporary storage...")

    try:
        print(f"Running {config['architecture']} on source: {source}")
        notebook, memory_bank, results = run_full_pipeline(resolved_source, config)

        # Keep the existing terminal output.
        print_notebook(notebook)
        print_results(results)
        print(f"\n=== Memory: {len(memory_bank)} caption entries ===")

        # Save a complete report separately.
        report = build_output(notebook, memory_bank, results, source)
        output_path = os.path.abspath(os.path.expanduser(args.output))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nResults saved to: {output_path}")

        if args.srt and results:
            srt_path = os.path.splitext(output_path)[0] + ".srt"
            export_to_srt(results, srt_path)
            print(f"Subtitles saved to: {srt_path}")
            
            if args.burn and source != 0:
                print("Burning subtitles into video via FFmpeg...")
                video_out = os.path.splitext(output_path)[0] + "_captioned.mp4"
                
                # FFmpeg requires escaping colons/slashes in the subtitle path
                escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
                
                cmd = f'ffmpeg -y -i "{resolved_source}" -vf "subtitles={escaped_srt}:force_style=\'FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2\'" -c:a copy "{video_out}" -loglevel error'
                if os.system(cmd) == 0:
                    print(f"Successfully created captioned video: {video_out}")
                else:
                    print("Failed to run FFmpeg. Make sure ffmpeg is installed.")

    finally:
        if temporary_path:
            cleanup_video_source(temporary_path)
            print("Temporary Google Drive video removed.")


if __name__ == "__main__":
    main()
