"""ShanghaiTech conversion: frame masks -> annotation rows -> evaluator labels, losslessly."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.evaluation.dataset import Annotation, frame_labels, load_annotations


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_shanghaitech")


@pytest.mark.parametrize("mask", [
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],          # starts at frame 0 and ends at the last frame
    [0, 1, 0, 1, 0, 1, 1],    # several separate intervals
    [1, 1, 1],
])
def test_mask_intervals_round_trip_through_evaluator(prepare, tmp_path, mask):
    mask = np.asarray(mask, dtype=np.uint8)
    ranges = prepare.intervals(mask)
    assert all(1 <= a <= b <= len(mask) for a, b in ranges)
    row = " ".join(["01_0001", "Anomaly"] + [str(v) for pair in ranges for v in pair])
    path = tmp_path / "annotations.txt"
    path.write_text(row + "\n")
    annotation = load_annotations(path)["01_0001"]
    assert annotation == Annotation("01_0001", "Anomaly", tuple(ranges))
    np.testing.assert_array_equal(frame_labels(annotation, len(mask), 24.0, "native"), mask)


@pytest.mark.parametrize("dataset", ["ucf", "xd", "shanghaitech", "avenue"])
def test_every_dataset_has_a_config_and_cli_choice(monkeypatch, dataset):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cli = importlib.import_module("evaluate")
    assert cli.DATASETS[dataset]
    assert (Path(__file__).resolve().parents[1] / "config/datasets" / f"{dataset}.yaml").exists()


def test_unknown_dataset_is_rejected(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cli = importlib.import_module("evaluate")
    with pytest.raises(SystemExit):
        cli.main(["--dataset", "kinetics", "--architecture", "action_default", "--gpus", "0", "--dry-run"])
