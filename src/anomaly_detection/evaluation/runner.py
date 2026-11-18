"""Complete-video benchmark evaluation for the window architectures (action, memory_novelty).

For each test video: score every window with the configured scorer, project the
window scores onto every original frame (temporal/projection.py), map them to
[0, 1] with scorer.unit, and compare with the frame labels. Memory is never
updated during testing.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np

from anomaly_detection.evaluation.dataset import NATIVE_END_TOLERANCE, frame_labels, manifest
from anomaly_detection.evaluation.metrics import compute_metrics, save_curves
from anomaly_detection.memory.store import file_digest
from anomaly_detection.reasoning.explain import explain_anomaly
from anomaly_detection.scoring import architecture, build_scorer
from anomaly_detection.temporal.projection import expand_events, expand_windows, project_window_scores

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(processor, bank, bank_metadata, test_videos, annotations, config,
             bank_path, annotation_path, output_dir, scorer=None):
    folder = Path(output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Results folder is not empty: {folder}; choose a new output directory")
    folder.mkdir(parents=True, exist_ok=True)
    predictions = folder / "predictions"
    predictions.mkdir()
    settings = config["evaluation"]
    review = settings.get("llm_review", False)
    annotation_fps = settings.get("annotation_fps", "native")
    constitution = [entry.text for entry in bank.entries] if bank is not None else []
    scorer = scorer or build_scorer(config, processor.encoder, bank, bank_path)
    threshold = float(scorer.threshold)
    temporal = config.get("temporal", {})
    windows = temporal.get("mode", "events") == "sliding_windows"
    if windows:
        score_expansion = (f"{temporal.get('score_pooling', 'mean')} of overlapping windows; smoothing "
                           f"{temporal.get('smoothing_seconds', 0)} s"
                           + (f", capped at video length / {temporal['smoothing_divisor']}"
                              if temporal.get("smoothing_divisor") else "")
                           + " (centered, nearest-padded boxcar)")
    else:
        score_expansion = "event value held until next event start, including final tail"
    bank_metadata = bank_metadata or {}
    metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture(config), "config": config,
        "memory_path": str(Path(bank_path).resolve()) if bank_path else None,
        "memory_sha256": file_digest(bank_path) if bank_path else None,
        "memory_created_utc": bank_metadata.get("created_utc"),
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": file_digest(annotation_path),
        "test_videos": manifest(test_videos), "annotation_fps": annotation_fps,
        "evaluation_grid": "all original video frames",
        "annotation_native_end_tolerance_frames": NATIVE_END_TOLERANCE,
        "score_expansion": score_expansion,
        "memory_updates_during_testing": False,
        "metric_protocol": "pooled frame-level ROC-AUC; fixed-threshold metrics; optional Gemma decisions",
        "reference_protocol": ("no memory bank" if bank is None else
                               "frozen source-domain memory; no target-domain memory adaptation"
                               if settings.get("frozen_reference_memory") else "target-domain normal reference memory"),
        "reference_memory_source_dataset": bank_metadata.get("config", {}).get("evaluation", {}).get("dataset"),
        "reference_normal_video_count": bank_metadata.get("normal_video_count", 0),
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
            results = scorer.score_video(video.events, video.frame_count, video.fps)
            window_scores, decisions, event_records = [], [], []
            for event, result in zip(video.events, results):
                result.is_anomaly = bool(scorer.unit(result.anomaly_score) > threshold)
                raw_decision = result.is_anomaly
                if review and raw_decision:
                    result = explain_anomaly(result, constitution, strict=True)
                window_scores.append(result.anomaly_score)
                decisions.append(result.is_anomaly)
                event_records.append({
                    "start_frame": event.start_frame, "last_sample_frame": event.end_frame,
                    "start_time": event.start_time, "stop_frame": event.stop_frame,
                    "caption": event.description, "components": result.component_scores,
                    "score": result.anomaly_score, "threshold_anomaly": raw_decision,
                    "final_anomaly": result.is_anomaly, "closest_match": result.closest_match,
                    "explanation": result.explanation,
                })
            window_scores = np.asarray(window_scores, dtype=float)
            if windows:
                projected = project_window_scores(video.events, window_scores, video.frame_count,
                                                  video.fps, temporal, clip=False)
            else:
                projected = expand_events(video.events, window_scores, video.frame_count)
            scores = np.asarray(scorer.unit(projected), dtype=float)
            expand = expand_windows if windows else expand_events
            reviewed = ((expand(video.events, np.asarray(decisions, dtype=float), video.frame_count) > 0.5)
                        if review else scores > threshold).astype(np.uint8)
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
            "architecture": architecture(config),
            "videos_evaluated": len(test_videos),
            "normal_videos": sum(not a.ranges for a in annotations.values()),
            "anomalous_videos": sum(bool(a.ranges) for a in annotations.values()),
            "normal_memory_videos": bank_metadata.get("normal_video_count", 0),
            "memory_entries": len(bank) if bank is not None else 0,
            "overall": compute_metrics(labels, scores, threshold,
                                       np.concatenate(reviewed_all) if review else None),
            "per_video": per_video,
            "notes": [
                "ROC-AUC uses continuous scores before Gemma verdict review, not binary flags.",
                "AUC is pooled across all frames, not averaged across videos.",
                "Single-class videos have null ROC-AUC and AP.",
                "The supplied dataset and split size are recorded in run.json.",
            ],
        }
        if getattr(getattr(scorer, "judge", None), "calibration", None):
            summary["judge_calibration"] = scorer.judge.calibration
        save_curves(labels, scores, folder, settings.get("dataset", "UCF-Crime"))
        write_json(folder / "summary.json", summary)
        metadata.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / "run.json", metadata)
        return summary
    except Exception as error:
        metadata.update(status="failed", failed_video=current_name, error=str(error))
        write_json(folder / "run.json", metadata)
        raise
