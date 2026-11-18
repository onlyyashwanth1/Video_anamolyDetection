#!/usr/bin/env python3
"""Convert official ShanghaiTech Campus data into the UCF-style evaluation layout.

Input (official release): training/videos/*.avi (330 normal videos),
testing/frames/<video>/<NNN>.jpg (107 videos) and testing/test_frame_mask/<video>.npy.

Output:
  Normal_Videos/            symlink to the official training videos (unchanged)
  Testing_Video/<video>.avi lossless FFV1 rebuild of every test JPEG frame
  Temporal_Anomaly_Annotation.txt  one-based inclusive native-frame intervals
  manifest.json             provenance, per-video FPS and verification results

Test frames carry no frame rate, so each test video uses the majority FPS of the
training videos from the same camera scene (24 or 25). Every converted video is
decoded again and must match the JPEG frames pixel for pixel, and every label
row must reproduce the official frame mask through the evaluator's own
frame_labels(). No labels are used for anything except writing the annotation.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anomaly_detection.evaluation.dataset import Annotation, frame_labels  # noqa: E402


def intervals(mask):
    """Zero-based mask -> one-based inclusive (start, end) runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a) + 1, int(b)) for a, b in zip(starts, stops)]


def scene_fps(training):
    by_scene = defaultdict(Counter)
    for path in sorted(training.glob("*.avi")):
        cap = cv2.VideoCapture(str(path))
        try:
            by_scene[path.stem[:2]][round(cap.get(cv2.CAP_PROP_FPS), 3)] += 1
        finally:
            cap.release()
    return {scene: counts.most_common(1)[0][0] for scene, counts in by_scene.items()}, by_scene


def convert(task):
    name, frame_dir, mask_path, fps, output = task
    output = Path(output)
    mask = np.load(mask_path)
    files = sorted(Path(frame_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(len(files))) or len(files) != len(mask):
        raise ValueError(f"{name}: frame files and mask disagree")
    frames_sha = hashlib.sha256()
    if not output.exists():
        temporary = output.with_name(output.stem + ".writing.avi")
        first = cv2.imread(str(files[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot write FFV1")
        try:
            for path in files:
                image = cv2.imread(str(path))
                if image is None or image.shape[:2] != (height, width):
                    raise ValueError(f"{name}: unreadable or resized frame {path.name}")
                writer.write(image)
        finally:
            writer.release()
        os.replace(temporary, output)
    # Verify the finished file exactly as the evaluator will read it.
    cap = cv2.VideoCapture(str(output))
    try:
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        declared_fps = cap.get(cv2.CAP_PROP_FPS)
        decoded = 0
        for path in files:
            ok, image = cap.read()
            if not ok:
                break
            if not np.array_equal(image, cv2.imread(str(path))):
                raise ValueError(f"{name}: decoded frame {decoded} differs from {path.name}")
            frames_sha.update(image.tobytes())
            decoded += 1
        extra = cap.read()[0]
    finally:
        cap.release()
    if decoded != len(files) or declared != len(files) or extra or abs(declared_fps - fps) > 1e-6:
        raise ValueError(f"{name}: decoded {decoded}, declared {declared}, expected {len(files)} frames")
    ranges = intervals(mask)
    labels = frame_labels(Annotation(name, "Anomaly", tuple(ranges)), len(mask), fps, "native")
    if not np.array_equal(labels, mask.astype(np.uint8)):
        raise ValueError(f"{name}: annotation round trip differs from official mask")
    return dict(name=name, frames=len(files), fps=fps, anomalous_frames=int(mask.sum()),
                intervals=ranges, mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                decoded_frames_sha256=frames_sha.hexdigest(), bytes=output.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="Extracted official 'shanghaitech' folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.output.resolve()
    training, frames, masks = raw / "training/videos", raw / "testing/frames", raw / "testing/test_frame_mask"
    test_names = sorted(p.name for p in frames.iterdir() if p.is_dir())
    if len(list(training.glob("*.avi"))) != 330 or len(test_names) != 107:
        raise ValueError("Expected the official 330 training / 107 test videos")
    if sorted(p.stem for p in masks.glob("*.npy")) != test_names:
        raise ValueError("Frame-mask files do not match test frame folders")
    fps_by_scene, fps_counts = scene_fps(training)
    (out / "Testing_Video").mkdir(parents=True, exist_ok=True)
    normal = out / "Normal_Videos"
    if normal.is_symlink():
        if normal.resolve() != training:
            raise ValueError(f"{normal} points somewhere else")
    elif normal.exists():
        raise FileExistsError(f"{normal} exists and is not the expected symlink")
    else:
        normal.symlink_to(training, target_is_directory=True)
    tasks = [(n, str(frames / n), str(masks / (n + ".npy")), fps_by_scene[n[:2]],
              str(out / "Testing_Video" / (n + ".avi"))) for n in test_names]
    cv2.setNumThreads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = []
        for record in pool.map(convert, tasks):
            records.append(record)
            print(f"[{len(records)}/{len(tasks)}] {record['name']}: {record['frames']} frames "
                  f"@ {record['fps']} fps, {len(record['intervals'])} anomalous interval(s)", flush=True)
    lines = [" ".join([r["name"], "Anomaly"] + [str(v) for pair in r["intervals"] for v in pair])
             for r in records]
    annotation = out / "Temporal_Anomaly_Annotation.txt"
    annotation.write_text("\n".join(lines) + "\n")
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), source=str(raw),
        source_release="ShanghaiTech Campus official layout (HF mirror 123zdm/ShanghaiTech)",
        protocol="original one-class split: 330 normal training videos, 107 test videos",
        test_encoding="FFV1 lossless AVI rebuilt from test JPEG frames; verified pixel-identical on decode",
        test_fps_rule="majority FPS of same-scene training videos", fps_by_scene=fps_by_scene,
        training_fps_counts={k: dict(v) for k, v in sorted(fps_counts.items())},
        annotation_format="name category start end ...; one-based inclusive native frames",
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        test_videos=len(records), test_frames=sum(r["frames"] for r in records),
        anomalous_frames=sum(r["anomalous_frames"] for r in records),
        videos_without_anomaly=sum(not r["intervals"] for r in records), videos=records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {annotation} and {out / 'manifest.json'}: {manifest['test_frames']} test frames, "
          f"{manifest['anomalous_frames']} anomalous", flush=True)


if __name__ == "__main__":
    main()
