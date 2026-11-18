#!/usr/bin/env python3
"""Score the frozen bank-free shared model on an external dataset's completed run.

The source run (for example the ShanghaiTech memory+action run) supplies the
cached CLIP window embeddings, the Gemma window captions and the native-frame
labels. Nothing is fitted here: prompts, temperatures, fusion weights, temporal
projection and threshold come unchanged from the configuration. Every test video
belongs to a single external test partition; there is no validation subset.

Before scoring, the source run's own action components are recomputed from the
same cached inputs with the production ActionScorer. They must match the stored
values, which checks window/caption alignment and text-encoder equivalence.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from anomaly_detection.evaluation.metrics import compute_metrics, save_curves  # noqa: E402
from anomaly_detection.inference.action_scoring import ActionScorer  # noqa: E402
from anomaly_detection.inference.frame_projection import project_window_scores  # noqa: E402
from evaluate_shared_model import digest_file, feature_contract, write_json  # noqa: E402
from prepare_shared_prompt_features import load_windows  # noqa: E402

SELF_CHECK_TOLERANCE = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/shared_best.yaml")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Device for the frozen CLIP text tower")
    parser.add_argument("--save-curves", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (config["inference"]["scoring_mode"] != "action"
            or config.get("memory", {}).get("representation") != "none"):
        raise ValueError("This evaluator requires a bank-free action-scoring configuration")
    source = args.source_run.resolve()
    source_run = json.loads((source / "run.json").read_text())
    if source_run["status"] != "complete":
        raise ValueError(f"Source run incomplete: {source}")
    source_config = source_run["config"]
    dataset = source_config["evaluation"]["dataset"]
    if feature_contract(config) != feature_contract(source_config):
        raise ValueError("Encoder/sampling/caption contract differs from the source run")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    predictions = output / "predictions"
    predictions.mkdir(parents=True)

    windows = load_windows(dataset, source)
    names, offsets = windows["names"].tolist(), windows["offsets"]
    images = windows["images"]
    from anomaly_detection.encoding.multi_scale_encoder import MultiScaleClipEncoder
    encoder = MultiScaleClipEncoder(config["encoder"]["model_name"], device=args.device)
    unique = list(dict.fromkeys(windows["captions"].tolist()))
    lookup = {text: i for i, text in enumerate(unique)}
    caption_vectors = encoder.encode_texts(unique, batch_size=128)
    texts = caption_vectors[[lookup[c] for c in windows["captions"]]]

    reproduced = ActionScorer(encoder, source_config["inference"].get("action", {})).score_vectors(images, texts)
    self_check = {
        "image_margin_max_abs_error": float(np.max(np.abs(
            reproduced["action_image_margin"] - windows["original_image_margin"]))),
        "text_margin_max_abs_error": float(np.max(np.abs(
            reproduced["action_text_margin"] - windows["original_text_margin"]))),
        "tolerance": SELF_CHECK_TOLERANCE}
    if max(self_check["image_margin_max_abs_error"], self_check["text_margin_max_abs_error"]) > SELF_CHECK_TOLERANCE:
        raise ValueError(f"Cannot reproduce the source run's stored action components: {self_check}")

    window_scores = np.asarray(
        ActionScorer(encoder, config["inference"].get("action", {})).score_vectors(images, texts)["action_score"],
        dtype=float)
    threshold = float(config["inference"]["anomaly_threshold"])
    config_sha = digest_file(args.config)
    metadata = dict(
        status="running", started_utc=datetime.now(timezone.utc).isoformat(), dataset=dataset,
        config=config, config_path=str(args.config.resolve()), config_sha256=config_sha,
        source_run=str(source), source_summary_sha256=digest_file(source / "summary.json"),
        feature_contract=feature_contract(config), self_check=self_check,
        production_scorer="ActionScorer.score_vectors + project_window_scores",
        evaluation_grid="all original video frames", normal_reference_memory_used=False,
        threshold_selected_during_this_evaluation=False, partition="external test (all videos)",
        unique_captions=len(unique),
        limitations=["Configuration was selected on UCF-Crime and XD-Violence only; "
                     "no setting was tuned on this dataset.",
                     "Centered smoothing, if configured, uses future context (offline evaluation)."])
    write_json(output / "run.json", metadata)
    labels_all, scores_all, per_video = [], [], {}
    for index, name in enumerate(names):
        a, b = map(int, offsets[index:index + 2])
        source_json = json.loads((source / "predictions" / (name + ".json")).read_text())
        events = source_json["events"]
        if len(events) != b - a:
            raise ValueError(f"Window count mismatch for {name}")
        with np.load(source / "predictions" / (name + ".npz"), allow_pickle=False) as z:
            labels, fps = z["labels"].copy(), float(z["fps"])
        if fps != source_json["fps"] or len(labels) != source_json["frame_count"]:
            raise ValueError(f"Native frame metadata mismatch for {name}")
        event_objects = [SimpleNamespace(start_frame=e["start_frame"], stop_frame=e["stop_frame"]) for e in events]
        scores = project_window_scores(event_objects, window_scores[a:b], len(labels), fps, config["temporal"])
        per_video[name] = dict(category=source_json["category"], **compute_metrics(labels, scores, threshold))
        np.savez_compressed(predictions / (name + ".npz"), scores=scores, labels=labels,
                            final_decisions=(scores > threshold).astype(np.uint8), fps=np.asarray(fps),
                            window_scores=window_scores[a:b], start_frame=windows["start_frame"][a:b],
                            stop_frame=windows["stop_frame"][a:b])
        labels_all.append(labels)
        scores_all.append(scores)
    labels, scores = np.concatenate(labels_all), np.concatenate(scores_all)
    original = json.loads((source / "summary.json").read_text())
    summary = dict(
        dataset=dataset, model=config.get("evaluation", {}).get("model_variant"), config_sha256=config_sha,
        videos_evaluated=len(names), frames=int(len(labels)), anomalous_frames=int(labels.sum()),
        overall=compute_metrics(labels, scores, threshold),
        source_run_overall={"model": source_config["evaluation"].get("model_variant"),
                            **original["overall"]},
        self_check=self_check, per_video=per_video,
        notes=["AUC/AP pooled over all native frames of all test videos (micro).",
               "Threshold metrics use the frozen UCF/XD-selected threshold."])
    if args.save_curves:
        save_curves(labels, scores, output, dataset)
    write_json(output / "summary.json", summary)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", metadata)
    print(f"{dataset} shared model: AUC={summary['overall']['roc_auc']:.4f} "
          f"AP={summary['overall']['average_precision']:.4f}  (source run "
          f"{summary['source_run_overall']['model']}: AUC={original['overall']['roc_auc']:.4f})", flush=True)


if __name__ == "__main__":
    main()
