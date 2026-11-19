"""Collect the latest complete evaluation of every dataset x architecture into one table.

Reads outputs/evaluations/<dataset>/<architecture>/<run>/ (run.json status "complete"),
recomputes pooled frame ROC-AUC / AP from the per-video predictions, and for UCF and XD
also on the study's grouped validation / evaluation partitions (seed 42, 20 / 80).

Usage: python scripts/collect_results.py [--out outputs/results/architectures]
Writes <out>/results.json and <out>/results.md.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "outputs/evaluations"
SPLITS = ROOT / "outputs/results/score_ablations_20261026/ablations.json"
DATASETS = ["ucf", "xd", "shanghaitech", "avenue"]
ARCHITECTURES = ["action_default", "shared_best", "memory_b", "b_plus_j"]
SPLIT_NAMES = {"ucf": "UCF-Crime", "xd": "XD-Violence"}
PRIMARY = {"ucf": "auc", "xd": "ap", "shanghaitech": "auc", "avenue": "auc"}

# Earlier numbers (masked features for the action models, research scripts for B / B+J).
REFERENCE = {
    ("ucf", "action_default"): {"auc": 77.57, "note": "masked"},
    ("xd", "action_default"): {"ap": 53.41, "note": "masked"},
    ("ucf", "shared_best"): {"auc": 83.12, "note": "masked"},
    ("xd", "shared_best"): {"ap": 71.76, "note": "masked"},
    ("ucf", "memory_b"): {"auc": 72.58, "note": "study"},
    ("xd", "memory_b"): {"ap": 54.00, "note": "study"},
    ("shanghaitech", "memory_b"): {"auc": 62.1, "note": "study"},
    ("avenue", "memory_b"): {"auc": 61.3, "note": "study"},
    ("ucf", "b_plus_j"): {"auc": 78.07, "note": "study"},
    ("avenue", "b_plus_j"): {"auc": 63.8, "note": "study"},
    ("shanghaitech", "shared_best"): {"auc": 54.14, "note": "masked"},
    ("avenue", "shared_best"): {"auc": 54.9, "note": "masked"},
}


def latest_complete(dataset, architecture):
    folder = EVALS / dataset / architecture
    if not folder.is_dir():
        return None
    for run in sorted(folder.iterdir(), reverse=True):
        try:
            if json.loads((run / "run.json").read_text()).get("status") == "complete":
                return run
        except (OSError, json.JSONDecodeError):
            continue
    return None


def pooled(chunks):
    if not chunks:
        return None
    labels = np.concatenate([c[0] for c in chunks])
    scores = np.concatenate([c[1] for c in chunks])
    if len(np.unique(labels)) < 2:
        return None
    return {"frames": int(len(labels)),
            "auc": round(100 * float(roc_auc_score(labels, scores)), 3),
            "ap": round(100 * float(average_precision_score(labels, scores)), 3)}


def partitions(dataset):
    if dataset not in SPLIT_NAMES or not SPLITS.exists():
        return {}
    split = {r["dataset"]: r for r in json.loads(SPLITS.read_text())}[SPLIT_NAMES[dataset]]
    out = {}
    for name in split["diagnostic_validation_videos"]:
        out[name] = "validation"
    for name in split["diagnostic_evaluation_videos"]:
        out[name] = "evaluation"
    return out


def evaluate_run(run, dataset):
    part_of = partitions(dataset)
    groups = {"all": [], "validation": [], "evaluation": []}
    for npz in sorted((run / "predictions").glob("*.npz")):
        with np.load(npz) as z:
            chunk = (z["labels"].astype(np.int8), z["scores"].astype(np.float64))
        groups["all"].append(chunk)
        if npz.stem in part_of:
            groups[part_of[npz.stem]].append(chunk)
    result = {k: pooled(v) for k, v in groups.items()}
    summary = json.loads((run / "summary.json").read_text())
    result["videos"] = summary.get("videos_evaluated")
    if summary.get("judge_calibration"):
        result["judge_calibration"] = {k: v for k, v in summary["judge_calibration"].items()
                                       if isinstance(v, (int, float, str))}
    return result


def fmt(metric):
    return "—" if metric is None else f"{metric:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/results/architectures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            run = latest_complete(dataset, architecture)
            row = {"dataset": dataset, "architecture": architecture,
                   "run": str(run.relative_to(ROOT)) if run else None,
                   "reference": REFERENCE.get((dataset, architecture))}
            if run:
                row.update(evaluate_run(run, dataset))
            rows.append(row)
            print(dataset, architecture, run.name if run else "missing", flush=True)

    stamp = datetime.now().isoformat(timespec="seconds")
    (args.out / "results.json").write_text(json.dumps({"generated": stamp, "rows": rows}, indent=2))

    lines = [f"# Architecture results (generated {stamp})", "",
             "Pooled frame-level metrics from `outputs/evaluations/`, unmasked features, restructured code.",
             "UCF / XD partitions: grouped 20% validation / 80% evaluation split (seed 42) of the earlier study.",
             "", "## Primary metric (UCF, ShanghaiTech, Avenue: ROC-AUC; XD: AP)", "",
             "| Architecture | UCF AUC | XD AP | ShanghaiTech AUC | Avenue AUC |", "|---|---|---|---|---|"]
    by_key = {(r["dataset"], r["architecture"]): r for r in rows}
    for architecture in ARCHITECTURES:
        cells = []
        for dataset in DATASETS:
            r = by_key[(dataset, architecture)]
            value = r.get("all", {}) and r["all"][PRIMARY[dataset]] if r.get("all") else None
            ref = r["reference"]
            ref_text = f" (was {ref[PRIMARY[dataset]]}, {ref['note']})" if ref and PRIMARY[dataset] in ref else ""
            cells.append(fmt(value) + ref_text)
        lines.append(f"| {architecture} | " + " | ".join(cells) + " |")

    lines += ["", "## All metrics", "",
              "| Dataset | Architecture | AUC all | AP all | AUC val | AP val | AUC eval | AP eval | Run |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        get = lambda part, m: (r.get(part) or {}).get(m)
        lines.append(f"| {r['dataset']} | {r['architecture']} | {fmt(get('all', 'auc'))} | {fmt(get('all', 'ap'))} | "
                     f"{fmt(get('validation', 'auc'))} | {fmt(get('validation', 'ap'))} | "
                     f"{fmt(get('evaluation', 'auc'))} | {fmt(get('evaluation', 'ap'))} | "
                     f"{'`' + r['run'] + '`' if r['run'] else 'missing'} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
