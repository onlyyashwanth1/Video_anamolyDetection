"""Orchestrate complete-video evaluation against a fixed normal memory bank."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from anomaly_detection.evaluation.bank import file_digest
from anomaly_detection.evaluation.dataset import NATIVE_END_TOLERANCE, expand_events, frame_labels, manifest
from anomaly_detection.evaluation.metrics import compute_metrics, save_curves
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.reasoning.llm_reasoning import explain_anomaly

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(processor, bank, bank_metadata, test_videos, annotations, config,
             bank_path, annotation_path, output_dir):
    folder = Path(output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Results folder is not empty: {folder}; choose a new output directory")
    folder.mkdir(parents=True, exist_ok=True)
    predictions = folder / "predictions"
    predictions.mkdir()
    settings = config["evaluation"]
    threshold = float(config["inference"]["anomaly_threshold"])
    review = settings.get("llm_review", True)
    annotation_fps = settings.get("annotation_fps", "native")
    constitution = [entry.text for entry in bank.entries]
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "memory_path": str(Path(bank_path).resolve()),
        "memory_sha256": file_digest(bank_path), "memory_created_utc": bank_metadata["created_utc"],
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": file_digest(annotation_path),
        "test_videos": manifest(test_videos), "annotation_fps": annotation_fps,
        "evaluation_grid": "all original video frames",
        "annotation_native_end_tolerance_frames": NATIVE_END_TOLERANCE,
        "score_expansion": "event value held until next event start, including final tail",
        "memory_updates_during_testing": False,
        "metric_protocol": "pooled frame-level ROC-AUC; fixed-threshold metrics; optional Gemma decisions",
    }
    write_json(folder / "run.json", metadata)
    scores_all, labels_all, reviewed_all = [], [], []
    per_video = {}
    current_name = None
    try:
        for index, (name, path) in enumerate(test_videos.items(), 1):
            current_name = name
            LOGGER.info("Test [%d/%d] %s", index, len(test_videos), name)
            video = processor.process(str(path))
            labels = frame_labels(annotations[name], video.frame_count, video.fps, annotation_fps)
            event_scores, decisions, event_records = [], [], []
            for event in video.events:
                result = score_event(event, bank, threshold, encoder=processor.encoder)
                # Use the saved continuous score consistently for threshold reporting.
                result.is_anomaly = result.anomaly_score > threshold
                raw_decision = result.is_anomaly
                if review and raw_decision:
                    result = explain_anomaly(result, constitution, strict=True)
                event_scores.append(result.anomaly_score)
                decisions.append(result.is_anomaly)
                event_records.append({
                    "start_frame": event.start_frame, "last_sample_frame": event.end_frame,
                    "start_time": event.start_time, "caption": event.description,
                    "score": result.anomaly_score, "threshold_anomaly": raw_decision,
                    "final_anomaly": result.is_anomaly, "closest_match": result.closest_match,
                    "explanation": result.explanation,
                })
            scores = expand_events(video.events, np.asarray(event_scores, dtype=np.float32), video.frame_count)
            reviewed = expand_events(video.events, np.asarray(decisions, dtype=np.uint8), video.frame_count)
            per_video[name] = {"category": annotations[name].category,
                               **compute_metrics(labels, scores, threshold, reviewed if review else None)}
            np.savez_compressed(predictions / f"{name}.npz", scores=scores, labels=labels,
                                final_decisions=reviewed, fps=np.asarray(video.fps))
            write_json(predictions / f"{name}.json", {
                "video": name, "path": str(path), "fps": video.fps,
                "frame_count": video.frame_count, "category": annotations[name].category,
                "events": event_records, "metrics": per_video[name],
            })
            scores_all.append(scores)
            labels_all.append(labels)
            if review:
                reviewed_all.append(reviewed)
            del video
        labels, scores = np.concatenate(labels_all), np.concatenate(scores_all)
        if len(np.unique(labels)) != 2:
            raise ValueError("Overall ROC-AUC needs both normal and anomalous frames")
        summary = {
            "videos_evaluated": len(test_videos),
            "normal_videos": sum(not a.ranges for a in annotations.values()),
            "anomalous_videos": sum(bool(a.ranges) for a in annotations.values()),
            "normal_memory_videos": bank_metadata["normal_video_count"],
            "memory_entries": len(bank),
            "overall": compute_metrics(labels, scores, threshold,
                                       np.concatenate(reviewed_all) if review else None),
            "per_video": per_video,
            "notes": [
                "ROC-AUC uses continuous scores before Gemma verdict review, not binary flags.",
                "AUC is pooled across all frames, not averaged across videos.",
                "Single-class videos have null ROC-AUC and AP.",
                "This evaluates the project's normal-memory method, not a reproduction of either paper.",
                "The supplied dataset and split size are recorded in run.json.",
            ],
        }
        save_curves(labels, scores, folder, settings.get("dataset", "UCF-Crime"))
        write_json(folder / "summary.json", summary)
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / "run.json", metadata)
        return summary
    except Exception as error:
        metadata.update(status="failed", failed_video=current_name, error=str(error))
        write_json(folder / "run.json", metadata)
        raise
