import argparse
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.pipeline import load_config, run_full_pipeline
from anomaly_detection.utils.types import AnomalyResult


def format_full_report(
    video_source: str,
    constitution: List[str],
    memory_bank,
    results: List[AnomalyResult],
    anomaly_windows: List[Tuple[float, float]],
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("VIDEO ANOMALY DETECTION - PIPELINE & ACCURACY REPORT")
    lines.append("=" * 72)
    lines.append(f"Source: {video_source}\n")

    lines.append(f"=== 'Normal' Notebook (from first {len(constitution)} distinct patterns seen) ===")
    if not constitution:
        lines.append("  (empty - nothing was captioned during familiarisation)")
    for line in constitution:
        lines.append(f"  - {line}")

    lines.append(f"\n=== Live Event Results ({len(results)} events tested) ===")
    tp = tn = fp = fn = 0

    for res in results:
        event = res.event
        start, end = event.start_time, event.end_time
        score = res.anomaly_score
        pred_anomaly = res.is_anomaly

        actual_anomaly = any(
            max(start, win_start) < min(end, win_end)
            for win_start, win_end in anomaly_windows
        )

        if actual_anomaly and pred_anomaly:
            tp += 1
        elif not actual_anomaly and not pred_anomaly:
            tn += 1
        elif not actual_anomaly and pred_anomaly:
            fp += 1
        else:
            fn += 1

        tag = "[ANOMALY]" if pred_anomaly else "[normal ]"
        lines.append(f"{tag} t={start:.1f}-{end:.1f}s   score={score:.3f}  seen as: \"{event.description}\"")
        lines.append(f"           closest match in notebook: \"{res.closest_match}\"")
        if pred_anomaly and res.explanation:
            lines.append(f"           Explanation: {res.explanation}")

    lines.append(f"\n=== Final Memory Bank ({len(memory_bank.entries)} entries) ===")
    lines.append(f"MemoryBank({len(memory_bank.entries)} entries):")
    for entry in memory_bank.entries:
        lines.append(f"  - {entry.text} (seen {entry.occurrence_count}x)")

    # 1. Raw Vector Threshold Metrics
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # 2. Smart LLM-Gated Metrics (Filters out "insufficient evidence" false alarms)
    llm_tp = llm_tn = llm_fp = llm_fn = 0
    for res in results:
        event = res.event
        start, end = event.start_time, event.end_time
        actual_anomaly = any(
            max(start, win_start) < min(end, win_end)
            for win_start, win_end in anomaly_windows
        )
        is_llm_confirmed = res.is_anomaly and (
            res.explanation and "insufficient evidence" not in res.explanation.lower()
        )

        if actual_anomaly and is_llm_confirmed:
            llm_tp += 1
        elif not actual_anomaly and not is_llm_confirmed:
            llm_tn += 1
        elif not actual_anomaly and is_llm_confirmed:
            llm_fp += 1
        else:
            llm_fn += 1

    llm_prec = llm_tp / (llm_tp + llm_fp) if (llm_tp + llm_fp) > 0 else 0.0
    llm_rec = llm_tp / (llm_tp + llm_fn) if (llm_tp + llm_fn) > 0 else 0.0
    llm_f1 = 2 * (llm_prec * llm_rec) / (llm_prec + llm_rec) if (llm_prec + llm_rec) > 0 else 0.0
    llm_acc = (llm_tp + llm_tn) / total if total > 0 else 0.0

    lines.append("\n" + "=" * 72)
    lines.append("GROUND TRUTH EVALUATION METRICS")
    lines.append("=" * 72)
    lines.append(f"Ground Truth Anomaly Window(s): {anomaly_windows}")
    lines.append(f"Total Evaluated Events        : {total}\n")

    lines.append("--- [1] Raw Vector Similarity Metrics (Threshold Cutoff Only) ---")
    lines.append(f"True Positives (TP)           : {tp}  (Correct Alerts)")
    lines.append(f"True Negatives (TN)           : {tn}  (Correct Normal Passes)")
    lines.append(f"False Positives (FP)          : {fp}  (False Alarms)")
    lines.append(f"False Negatives (FN)          : {fn}  (Missed Anomalies)")
    lines.append(f"Accuracy                      : {accuracy * 100:.2f}%")
    lines.append(f"Precision                     : {precision * 100:.2f}%")
    lines.append(f"Recall                        : {recall * 100:.2f}%")
    lines.append(f"F1-Score                      : {f1 * 100:.2f}%\n")

    lines.append("--- [2] Smart LLM-Gated Metrics (LLM Filters Out 'Insufficient Evidence' False Alarms) ---")
    lines.append(f"True Positives (TP)           : {llm_tp}")
    lines.append(f"True Negatives (TN)           : {llm_tn}")
    lines.append(f"False Positives (FP)          : {llm_fp}  (False Alarms Remaining)")
    lines.append(f"False Negatives (FN)          : {llm_fn}")
    lines.append(f"Accuracy (LLM-Gated)          : {llm_acc * 100:.2f}%")
    lines.append(f"Precision (LLM-Gated)         : {llm_prec * 100:.2f}%")
    lines.append(f"Recall (LLM-Gated)            : {llm_rec * 100:.2f}%")
    lines.append(f"F1-Score (LLM-Gated)          : {llm_f1 * 100:.2f}%")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72 + "\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection Accuracy against Ground Truth.")
    parser.add_argument("--video", type=str, required=True, help="Path or URL to test video")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.txt",
        help="Output text file path to save report",
    )
    parser.add_argument(
        "--anomaly-start",
        type=float,
        nargs="+",
        help="Start time(s) of anomaly in seconds (e.g. --anomaly-start 88.0)",
    )
    parser.add_argument(
        "--anomaly-end",
        type=float,
        nargs="+",
        help="End time(s) of anomaly in seconds (e.g. --anomaly-end 103.0)",
    )

    args = parser.parse_args()

    if not args.anomaly_start or not args.anomaly_end or len(args.anomaly_start) != len(args.anomaly_end):
        print("Error: Please provide matching --anomaly-start and --anomaly-end timestamps.")
        sys.exit(1)

    anomaly_windows = list(zip(args.anomaly_start, args.anomaly_end))

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    print(f"Running pipeline on video: {args.video}...")
    print(f"Ground Truth Anomaly Time Window(s): {anomaly_windows}")

    constitution, memory_bank, results = run_full_pipeline(args.video, config)

    report = format_full_report(args.video, constitution, memory_bank, results, anomaly_windows)
    print(report)

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Saved Evaluation Report to: {output_path}]")


if __name__ == "__main__":
    main()