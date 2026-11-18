#!/usr/bin/env python3
"""ShanghaiTech follow-up analyses on saved features (no new Gemma/CLIP-vision work).

Part A (label-free, legitimate): threshold = a percentile of frame scores on
ShanghaiTech NORMAL training videos only. shared_best is bank-free, so all 330
training videos are used. The memory+action model would trivially match its own
memory entries, so it uses only the 66 held-out calibration videos that are not
in its memory. No anomaly labels are used to choose the threshold. AUC/AP do
not depend on a threshold and are unchanged by Part A.

Part B (ORACLE, test labels used): sweep the action/memory weight and the memory
image/text mix on the TEST set. This is an upper bound on what per-dataset weight
tuning could give; it is not a generalization result.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anomaly_detection.evaluation.bank import load_bank  # noqa: E402
from anomaly_detection.evaluation.dataset import discover_videos  # noqa: E402
from anomaly_detection.evaluation.metrics import compute_metrics  # noqa: E402
from anomaly_detection.evaluation.video_cache import VideoCache  # noqa: E402
from anomaly_detection.inference.action_scoring import ActionScorer  # noqa: E402
from anomaly_detection.inference.frame_projection import project_window_scores  # noqa: E402

PERCENTILES = (95.0, 99.0, 99.9)


def frame_scores(events, window_scores, frame_count, fps, temporal):
    return project_window_scores(events, np.asarray(window_scores, dtype=float), frame_count, fps, temporal)


def threshold_report(test_dir, normal_scores, threshold_frozen):
    labels, scores = [], []
    for path in sorted(Path(test_dir, "predictions").glob("*.npz")):
        with np.load(path) as z:
            labels.append(z["labels"])
            scores.append(z["scores"])
    labels, scores = np.concatenate(labels), np.concatenate(scores)
    out = {"frozen": compute_metrics(labels, scores, threshold_frozen)}
    for p in PERCENTILES:
        t = float(np.percentile(normal_scores, p))
        out[f"normal_p{p:g}"] = compute_metrics(labels, scores, t)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Completed ShanghaiTech memory+action run")
    parser.add_argument("--shared-run", type=Path, required=True, help="evaluate_shared_external.py output")
    parser.add_argument("--shared-config", type=Path, default=ROOT / "config/shared_best.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    config = run["config"]
    config["memory"]["search_device"] = "cpu"
    shared = yaml.safe_load(args.shared_config.read_text())
    cache = VideoCache(ROOT / config["evaluation"]["checkpoint_dir"], config)
    from anomaly_detection.encoding.multi_scale_encoder import MultiScaleClipEncoder
    encoder = MultiScaleClipEncoder(config["encoder"]["model_name"], device="cpu")
    normal = discover_videos(config["evaluation"]["normal_dir"])
    bank, bank_meta = load_bank(config["evaluation"]["memory_bank_path"], config, normal)
    calibration = {v["name"] for v in bank_meta["calibration_videos"]}
    in_memory = {v["name"] for v in bank_meta["reference_videos"]}
    if calibration & in_memory:
        raise ValueError("Calibration videos overlap memory videos")
    own_scorer = ActionScorer(encoder, config["inference"]["action"])
    shared_scorer = ActionScorer(encoder, shared["inference"]["action"])
    inference = config["inference"]
    w, miw = float(inference["action_weight"]), float(inference["memory_image_weight"])

    # ---- Part A: normal-video score distributions --------------------------------
    shared_normal, own_normal = [], []
    for i, (name, path) in enumerate(normal.items(), 1):
        video = cache.load(str(path))
        if video is None:
            raise FileNotFoundError(f"Missing cached normal video {name}")
        images = np.stack([e.average_embedding for e in video.events])
        texts = encoder.encode_texts([e.description for e in video.events], batch_size=128)
        s = shared_scorer.score_vectors(images, texts)["action_score"]
        shared_normal.append(frame_scores(video.events, s, video.frame_count, video.fps, shared["temporal"]))
        if name in calibration:
            action = own_scorer.score_vectors(images, texts)["action_score"]
            memory = np.array([bank.score_memory(im, tx, image_weight=miw, k=bank.top_k)["memory_score"]
                               for im, tx in zip(images, texts)])
            combined = (1 - w) * memory + w * action
            own_normal.append(frame_scores(video.events, combined, video.frame_count, video.fps,
                                           config.get("temporal", {})))
        if i % 50 == 0:
            print(f"normal videos scored: {i}/{len(normal)}", flush=True)
    part_a = {
        "shared_best": threshold_report(args.shared_run, np.concatenate(shared_normal),
                                        float(shared["inference"]["anomaly_threshold"])),
        "memory_plus_action": threshold_report(args.run, np.concatenate(own_normal),
                                               float(inference["anomaly_threshold"])),
        "normal_videos": {"shared_best": len(shared_normal), "memory_plus_action_calibration": len(own_normal)},
    }

    # ---- Part B: oracle weight sweep on test -------------------------------------
    videos = []
    for p in sorted((args.run / "predictions").glob("*.json")):
        d = json.loads(p.read_text())
        with np.load(p.with_suffix(".npz")) as z:
            labels = z["labels"]
        comp = {k: np.array([e["components"][k] for e in d["events"]])
                for k in ("action_score", "memory_image_score", "memory_text_score")}
        events = [type("E", (), dict(start_frame=e["start_frame"], stop_frame=e["stop_frame"]))()
                  for e in d["events"]]
        videos.append((events, comp, labels, d["frame_count"], d["fps"]))
    temporal = config.get("temporal", {})

    def pooled(aw, mw):
        L, S = [], []
        for events, c, labels, n, fps in videos:
            mem = mw * c["memory_image_score"] + (1 - mw) * c["memory_text_score"]
            S.append(frame_scores(events, (1 - aw) * mem + aw * c["action_score"], n, fps, temporal))
            L.append(labels)
        m = compute_metrics(np.concatenate(L), np.concatenate(S), 0.5)
        return m["roc_auc"], m["average_precision"]

    reproduced = pooled(w, miw)
    grid = []
    for aw in np.round(np.arange(0, 1.0001, 0.05), 2):
        for mw in np.round(np.arange(0, 1.0001, 0.25), 2):
            auc, ap = pooled(aw, mw)
            grid.append(dict(action_weight=float(aw), memory_image_weight=float(mw), roc_auc=auc,
                             average_precision=ap))
    best = max(grid, key=lambda r: r["roc_auc"])
    part_b = dict(warning="ORACLE: weights chosen with ShanghaiTech test labels; upper bound only.",
                  frozen=dict(action_weight=w, memory_image_weight=miw, roc_auc=reproduced[0],
                              average_precision=reproduced[1]),
                  best=best, grid=grid)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(dict(part_a=part_a, part_b=part_b), indent=2) + "\n")

    print("\nPart A — threshold from normal training videos only (AUC/AP unchanged):")
    for model, rows in part_a.items():
        if model == "normal_videos":
            continue
        for key, r in rows.items():
            t = r["threshold_metrics"]
            print(f"  {model:18s} {key:14s} thr={r['threshold']:.4f}  P={t['precision']*100:5.1f} "
                  f"R={t['recall']*100:5.1f} F1={t['f1']*100:5.1f}")
    print(f"\nPart B — ORACLE sweep (test labels). Frozen reproduced: AUC={reproduced[0]*100:.2f} "
          f"AP={reproduced[1]*100:.2f}")
    print(f"  best: action_weight={best['action_weight']}, memory_image_weight={best['memory_image_weight']}"
          f" -> AUC={best['roc_auc']*100:.2f} AP={best['average_precision']*100:.2f}")
    for r in grid:
        if r["memory_image_weight"] == best["memory_image_weight"] and r["action_weight"] in (0, .25, .5, .75, .95, 1):
            print(f"  action_weight={r['action_weight']:.2f}: AUC={r['roc_auc']*100:.2f}")


if __name__ == "__main__":
    main()
