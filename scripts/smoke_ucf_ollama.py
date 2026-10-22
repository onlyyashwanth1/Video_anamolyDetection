#!/usr/bin/env python3
"""Check real Ollama captions, or preprocess two short UCF videos before a full run."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import yaml
    from anomaly_detection.evaluation.multi_gpu import (
        _configure_gpu, cache_lock, finalize, pending_jobs, resolve_devices, run_workers,
    )
    from anomaly_detection.evaluation.video_cache import VideoCache
    from anomaly_detection.reasoning.ollama_runtime import resolve_model, ollama_servers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ucf_ollama.yaml")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)
    config = yaml.safe_load(args.config.read_text())
    resolve_model(config)
    devices = resolve_devices(args.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "outputs/smoke_checks" / stamp
    folder.mkdir(parents=True)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    settings = config["evaluation"]
    source = Path(settings["test_dir"]) / "RoadAccidents021_x264.mp4"
    if args.models_only:
        with ollama_servers(devices[:1], config, folder / "ollama"):
            _configure_gpu(devices[0], 4)
            from anomaly_detection.reasoning import local_gemma
            import cv2
            from PIL import Image
            local_gemma.configure(config, devices[0])
            local_gemma.ensure_ready()
            capture = cv2.VideoCapture(str(source))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError(f"Could not read a real frame from {source}")
            caption = local_gemma.generate_caption(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                                   "Describe the visible action in one short sentence.")
            summary = local_gemma.generate_text("Summarize this observation in one short sentence: " + caption, 60)
            result = {"check": "real image and text generation", "source": str(source),
                      "caption": caption, "summary": summary, "model": config["reasoning"]}
    else:
        sources = {source.stem: source}
        normal = Path(settings["normal_dir"]) / "Normal_Videos157_x264.mp4"
        sources[normal.stem] = normal
        cache_dir = ROOT / "outputs/checkpoints/ucf_ollama"
        with cache_lock(cache_dir):
            cache = VideoCache(cache_dir, config)
            jobs, resumed = pending_jobs(sources, cache)
            with ollama_servers(devices, config, folder / "ollama"):
                run_workers(devices, config, cache_dir, jobs, 4)
                from anomaly_detection.evaluation.dataset import load_annotations
                annotation_path = Path(settings["annotation_file"])
                finalize(devices[0], config, cache_dir, {normal.stem: normal},
                         {source.stem: source}, load_annotations(annotation_path),
                         folder / "smoke_memory.npz", annotation_path, folder / "evaluation",
                         "all", 4)
            records = []
            for name, path in sources.items():
                video = cache.load(path)
                if video is None:
                    raise RuntimeError(f"No checkpoint for {name}")
                records.append({"name": name, "frames": video.frame_count,
                                "events": len(video.events),
                                "captions": [event.description for event in video.events]})
            result = {"check": "two complete videos", "gpus": devices, "resumed": resumed,
                      "videos": records, "model": config["reasoning"],
                      "evaluation": str(folder / "evaluation")}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved smoke-test result: {folder / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
