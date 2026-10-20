"""Exercise the XD parallel CLI with real split parsing and no model loading."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def xd_cli(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cli = importlib.import_module("evaluate_ucf_multi_gpu")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root = tmp_path / "xd"
    for folder, names in [
        ("Normal_Videos", ["Movie.2001_label_A"]),
        ("Testing_Video", ["Other.2002_label_A", "Other.2002_label_B1"]),
    ]:
        (root / folder).mkdir(parents=True)
        for name in names:
            (root / folder / f"{name}.mp4").touch()
    (root / "train_split.txt").write_text("Movie.2001_label_A 0 0\n")
    (root / "test_split.txt").write_text("Other.2002_label_A 0 0\nOther.2002_label_B1 1 1\n")
    (root / "annotations.txt").write_text("Other.2002_label_B1.mp4 0 2\n")
    config = {
        "encoder": {"model_name": "fake"},
        "video": {"target_fps": 1},
        "segmentation": {},
        "familiarisation": {"min_caption_occurrences": 1},
        "inference": {"anomaly_threshold": .5},
        "evaluation": {"annotation_fps": 5, "llm_review": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    import cv2
    cap = Mock()
    cap.get.side_effect = lambda prop: 10 if prop == cv2.CAP_PROP_FPS else 50
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=cap))
    import anomaly_detection.evaluation.multi_gpu as parallel
    workers, finalize = Mock(), Mock()
    monkeypatch.setattr(parallel, "run_workers", workers)
    monkeypatch.setattr(parallel, "finalize", finalize)
    # main changes cwd for real runs; let pytest restore it afterwards.
    monkeypatch.chdir(tmp_path)
    args = ["--config", str(config_path), "--dataset-root", str(root), "--gpus", "0,1"]
    return cli, root, args, workers, finalize


@pytest.mark.parametrize("stage,expected_jobs", [("all", 3), ("build-memory", 1)])
def test_xd_worker_inputs_annotations_and_isolated_outputs(xd_cli, stage, expected_jobs):
    cli, root, args, workers, finalize = xd_cli
    cli.main("XD-Violence", args + ["--stage", stage])
    devices, config, cache_dir, jobs, threads = workers.call_args.args
    assert devices == ["0", "1"]
    assert len(jobs) == expected_jobs
    assert cache_dir.name == "xd_violence_multi_gpu"
    settings = config["evaluation"]
    assert settings["dataset"] == "XD-Violence"
    assert settings["annotation_fps"] == "native"
    assert settings["llm_review"] is False
    assert settings["annotation_source_indices"] == "zero-based inclusive"
    parameters = finalize.call_args.args
    normal, testing, annotations = parameters[3:6]
    assert set(normal) == {"Movie.2001_label_A"}
    assert len(testing) == 2
    assert annotations["Other.2002_label_B1"].ranges == ((1, 3),)
    assert annotations["Other.2002_label_A"].ranges == ()
    assert parameters[6].name == "xd_violence_multi_gpu.npz"
    assert parameters[7] == root / "annotations.txt"
    assert parameters[8].parent.name == "xd_violence_multi_gpu_evaluation"
    assert parameters[9] == stage


def test_xd_dry_run_never_starts_workers_or_writes_checkpoints(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    cli.main("XD-Violence", args + ["--dry-run"])
    workers.assert_not_called()
    finalize.assert_not_called()
    assert not (root.parent / "outputs").exists()


def test_xd_rejects_invalid_annotations_before_worker_launch(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    (root / "annotations.txt").write_text("")
    with pytest.raises(ValueError, match="annotations do not match"):
        cli.main("XD-Violence", args)
    workers.assert_not_called()
    finalize.assert_not_called()


def test_xd_rejects_individual_ucf_path_overrides(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    with pytest.raises(SystemExit) as error:
        cli.main("XD-Violence", args + ["--normal-dir", str(root / "Normal_Videos")])
    assert error.value.code == 2
    workers.assert_not_called()
