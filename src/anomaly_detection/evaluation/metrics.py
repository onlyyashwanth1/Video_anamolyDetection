"""Full-frame metrics from continuous scores; thresholds are never fit on test labels."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, precision_recall_curve, roc_auc_score, roc_curve,
)


def decision_metrics(labels, decisions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, decisions, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "accuracy": float(accuracy_score(labels, decisions)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def compute_metrics(labels, scores, threshold, reviewed=None):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Expected nonempty aligned binary labels and finite scores in [0,1]")
    both_classes = len(np.unique(labels)) == 2
    result = {
        "frames": len(labels), "anomalous_frames": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "threshold": threshold,
        "threshold_metrics": decision_metrics(labels, scores > threshold),
    }
    if reviewed is not None:
        result["gemma_reviewed_metrics"] = decision_metrics(labels, reviewed)
    return result


def save_curves(labels, scores, folder, dataset="UCF-Crime"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    np.savez_compressed(folder / "curves.npz", fpr=fpr, tpr=tpr,
                        roc_thresholds=roc_thresholds, precision=precision,
                        recall=recall, pr_thresholds=pr_thresholds)
    for filename, x, y, xlabel, ylabel in [
        ("roc.png", fpr, tpr, "False positive rate", "True positive rate"),
        ("precision_recall.png", recall, precision, "Recall", "Precision"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1),
               title=f"{dataset}: all evaluated frames")
        fig.savefig(folder / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
