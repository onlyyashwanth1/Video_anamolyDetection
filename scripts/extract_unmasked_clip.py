#!/usr/bin/env python3
"""Masking ablation: recompute window CLIP embeddings with the object mask disabled.

Matches the cached pipeline except for the masking step: identical 1 FPS frame
sampling (VideoStream rule), the same two-resolution MultiScaleClipEncoder,
and identical 4 s / 1 s sliding windows (build_sliding_windows). Gemma captions
never see masked frames, so they are unaffected and are not recomputed.

Frames are encoded in batches; --check-masked re-runs the masked path for a few
videos to measure the reproduction noise floor against the cache.

Outputs one NPZ per video: embeddings (windows x 768), start/stop frames.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCES = {
    "ucf": dict(run=ROOT / "outputs/ucf_expanded95_evaluation/20261022T192245Z",
                bank=ROOT / "outputs/memory/ucf_visual_text_v1.npz"),
    "xd": dict(run=ROOT / "outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z",
               bank=ROOT / "outputs/memory/xd_visual_text_20261022T192245Z.npz"),
    "sht": dict(run=ROOT / "outputs/shanghaitech_multi_gpu_evaluation/20261107T140236321294Z",
                bank=ROOT / "outputs/memory/shanghaitech_visual_text_v1.npz"),
}


def manifest(keys):
    jobs = []
    if "avenue" in keys:
        runs = sorted((ROOT / "outputs/avenue_multi_gpu_evaluation").glob("*/summary.json"))
        if not runs:
            raise FileNotFoundError("Run the Avenue evaluation first")
        SOURCES["avenue"] = dict(run=runs[-1].parent, bank=ROOT / "outputs/memory/avenue_visual_text_v1.npz")
    for key in keys:
        run = json.loads((SOURCES[key]["run"] / "run.json").read_text())
        for video in run["test_videos"]:
            jobs.append(dict(dataset=key, split="test", name=video["name"], path=video["path"],
                             bytes=video["bytes"]))
        with np.load(SOURCES[key]["bank"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
        for video in meta["reference_videos"]:
            jobs.append(dict(dataset=key, split="bank", name=video["name"], path=video["path"],
                             bytes=video["bytes"]))
    return sorted(jobs, key=lambda j: -j["bytes"])


def encode_batch(encoder, images, batch=64):
    import torch
    from PIL import Image
    from anomaly_detection.encoding.clip_encoder import _unwrap_features
    base = encoder.base_encoder
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            pil = [Image.fromarray(im) for im in images[i:i + batch]]
            small = [Image.fromarray(np.array(encoder.resizer(p))) for p in pil]
            fused = 0
            for group in (pil, small):
                inputs = base.processor(images=group, return_tensors="pt").to(base.device)
                f = _unwrap_features(base.model.get_image_features(**inputs))
                fused = fused + (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
            fused = fused / 2.0
            out.append(fused / np.linalg.norm(fused, axis=1, keepdims=True))
    return np.concatenate(out)


def process(job, encoder, masker, output):
    import cv2
    from anomaly_detection.ingestion.video_stream import VideoStream
    from anomaly_detection.segmentation.sliding_windows import build_sliding_windows
    from anomaly_detection.utils.types import EncodedFrame, Frame
    stream = VideoStream(job["path"], target_fps=1.0)
    cap = stream.cap
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idx = [], 0
    try:
        while True:
            if idx % stream.frame_interval == 0:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(Frame(frame_number=idx, timestamp_sec=idx / stream.native_fps,
                                    image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            elif not cap.grab():
                break
            idx += 1
    finally:
        stream.close()
    if masker is not None:
        masker.reset_tracking()
        frames = [masker.mask_frame(f) for f in frames]
    vectors = encode_batch(encoder, [f.image for f in frames])
    encoded = [EncodedFrame(frame=f, embedding=v) for f, v in zip(frames, vectors)]
    windows = build_sliding_windows(encoded, frame_count, stream.native_fps, 4.0, 1.0)
    temporary = output.with_name(output.name + ".writing.npz")
    np.savez(temporary, embeddings=np.stack([w.average_embedding for w in windows]).astype(np.float32),
             start_frame=np.asarray([w.start_frame for w in windows]),
             stop_frame=np.asarray([w.stop_frame for w in windows]),
             frame_count=np.asarray(frame_count), fps=np.asarray(stream.native_fps),
             sampled_frames=np.asarray(len(frames)))
    os.replace(temporary, output)


def worker(device, jobs, out_root, masked, done):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    from anomaly_detection.encoding.multi_scale_encoder import MultiScaleClipEncoder
    encoder = MultiScaleClipEncoder("openai/clip-vit-large-patch14", device="cuda:0")
    masker = None
    if masked:
        from anomaly_detection.masking.object_masking import SpatialObjectMasker
        os.chdir(ROOT)  # yolov8s-worldv2.pt lives in the repository root
        masker = SpatialObjectMasker(strict=True)
    while True:
        try:
            job = jobs.get(timeout=1)
        except Exception:
            return
        output = out_root / job["dataset"] / job["split"] / (job["name"] + ".npz")
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                process(job, encoder, masker, output)
            except Exception as error:  # report and continue; summary lists failures
                done.put((job["name"], f"FAILED: {error!r}"))
                continue
        done.put((job["name"], "ok"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="ucf,xd,sht")
    parser.add_argument("--gpus", default="0,1,2,7")
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/features/unmasked_clip_v1")
    parser.add_argument("--check-masked", type=int, default=0,
                        help="Instead: recompute N test videos WITH masking (noise-floor check)")
    args = parser.parse_args()
    jobs = manifest(args.datasets.split(","))
    out_root = args.output
    if args.check_masked:
        tests = [j for j in jobs if j["split"] == "test"]
        rng = np.random.default_rng(0)
        jobs = [tests[i] for i in rng.choice(len(tests), args.check_masked, replace=False)]
        out_root = args.output.parent / (args.output.name + "_masked_check")
    ctx = mp.get_context("spawn")
    queue, done = ctx.Queue(), ctx.Queue()
    for job in jobs:
        queue.put(job)
    devices = args.gpus.split(",")
    procs = [ctx.Process(target=worker, args=(d, queue, out_root, bool(args.check_masked), done))
             for d in devices for _ in range(args.workers_per_gpu)]
    for p in procs:
        p.start()
    started, failures = time.time(), []
    for i in range(1, len(jobs) + 1):
        name, status = done.get()
        if status != "ok":
            failures.append((name, status))
            print(name, status, flush=True)
        if i % 100 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} videos, {time.time() - started:.0f}s, failures {len(failures)}", flush=True)
    for p in procs:
        p.join()
    (out_root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__":
    main()
