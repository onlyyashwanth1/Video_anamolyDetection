#!/usr/bin/env python3
"""Convert CUHK Avenue (frames + avenue.mat) into the UCF-style evaluation layout.

Input: avenue/training/frames/<NN>/<NNNN>.jpg (16 normal videos),
       avenue/testing/frames/<NN>/<NNNN>.jpg (21 videos), avenue/avenue.mat
       (gt{i} = [starts; ends], one-based inclusive frame intervals, as used by the
       standard Avenue evaluation code).
Output: Normal_Videos/train_<NN>.avi and Testing_Video/test_<NN>.avi (lossless FFV1 at 25 FPS,
        verified pixel-identical on decode), Temporal_Anomaly_Annotation.txt, manifest.json.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anomaly_detection.evaluation.dataset import Annotation, frame_labels  # noqa: E402

FPS = 25.0


def write_video(task):
    frame_dir, output = Path(task[0]), Path(task[1])
    files = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(len(files))):
        raise ValueError(f"{frame_dir}: frame numbering is not contiguous from 0")
    if not output.exists():
        temporary = output.with_name(output.stem + ".writing.avi")
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (w, h))
        try:
            for p in files:
                img = cv2.imread(str(p))
                if img is None or img.shape[:2] != (h, w):
                    raise ValueError(f"bad frame {p}")
                writer.write(img)
        finally:
            writer.release()
        os.replace(temporary, output)
    cap = cv2.VideoCapture(str(output))
    try:
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        for p in files:
            ok, img = cap.read()
            if not ok or not np.array_equal(img, cv2.imread(str(p))):
                raise ValueError(f"{output.name}: decoded frame {decoded} differs")
            decoded += 1
        extra = cap.read()[0]
    finally:
        cap.release()
    if declared != len(files) or decoded != len(files) or extra:
        raise ValueError(f"{output.name}: frame count mismatch")
    return output.stem, len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="Extracted 'avenue' folder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.output.resolve()
    train = sorted(p for p in (raw / "training/frames").iterdir() if p.is_dir())
    test = sorted(p for p in (raw / "testing/frames").iterdir() if p.is_dir())
    if len(train) != 16 or len(test) != 21:
        raise ValueError("Expected 16 training and 21 test videos")
    gt = scipy.io.loadmat(raw / "avenue.mat")["gt"]
    if gt.shape != (1, 21):
        raise ValueError("Unexpected avenue.mat layout")
    (out / "Normal_Videos").mkdir(parents=True, exist_ok=True)
    (out / "Testing_Video").mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(out / "Normal_Videos" / f"train_{p.name}.avi")) for p in train]
    tasks += [(str(p), str(out / "Testing_Video" / f"test_{p.name}.avi")) for p in test]
    with ProcessPoolExecutor(max_workers=16) as pool:
        counts = dict(pool.map(write_video, tasks))
    lines, records = [], []
    for i, p in enumerate(test):
        name = f"test_{p.name}"
        events = np.asarray(gt[0, i]).reshape(2, -1)
        ranges = [(int(s), int(e)) for s, e in zip(events[0], events[1])]
        n = counts[name]
        if any(not 1 <= s <= e <= n for s, e in ranges):
            raise ValueError(f"{name}: interval outside 1..{n}")
        labels = frame_labels(Annotation(name, "Anomaly", tuple(ranges)), n, FPS, "native")
        expected = np.zeros(n, np.uint8)
        for s, e in ranges:
            expected[s - 1:e] = 1  # standard Avenue evaluation convention
        if not np.array_equal(labels, expected):
            raise ValueError(f"{name}: annotation round trip failed")
        lines.append(" ".join([name, "Anomaly"] + [str(v) for r in ranges for v in r]))
        records.append(dict(name=name, frames=n, intervals=ranges, anomalous_frames=int(expected.sum())))
    ann = out / "Temporal_Anomaly_Annotation.txt"
    ann.write_text("\n".join(lines) + "\n")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(raw),
                    source_release="CUHK Avenue frames + avenue.mat (HF mirror 123zdm/avenue)", fps=FPS,
                    encoding="FFV1 lossless AVI rebuilt from JPEG frames; verified pixel-identical",
                    training_videos=len(train), training_frames=sum(counts[f"train_{p.name}"] for p in train),
                    test_videos=len(test), test_frames=sum(r["frames"] for r in records),
                    anomalous_frames=sum(r["anomalous_frames"] for r in records),
                    annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest(), videos=records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"train {manifest['training_videos']} videos / {manifest['training_frames']} frames; test "
          f"{manifest['test_videos']} / {manifest['test_frames']} frames, {manifest['anomalous_frames']} anomalous")


if __name__ == "__main__":
    main()
