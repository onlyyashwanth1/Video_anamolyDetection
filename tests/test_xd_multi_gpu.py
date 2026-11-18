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


def test_xd_frozen_memory_reuses_source_and_only_processes_test(xd_cli, monkeypatch, tmp_path):
    cli, root, args, workers, finalize = xd_cli
    import anomaly_detection.evaluation.bank as banks
    bank = tmp_path / "source.npz"
    bank.touch()
    loader = Mock(return_value=(None, {"normal_videos": [
        {"name": "UCF_normal", "path": str(tmp_path / "UCF_normal.mp4")}]}))
    monkeypatch.setattr(banks, "load_bank", loader)
    cli.main("XD-Violence", args + ["--stage", "evaluate", "--frozen-reference-memory",
                                    "--memory-bank-path", str(bank)])
    assert loader.call_args.args[2] is None
    assert len(workers.call_args.args[3]) == 2
    assert finalize.call_args.args[1]["evaluation"]["frozen_reference_memory"] is True


def test_xd_frozen_memory_rejects_reference_test_overlap(xd_cli, monkeypatch, tmp_path):
    cli, root, args, workers, finalize = xd_cli
    import anomaly_detection.evaluation.bank as banks
    bank = tmp_path / "source.npz"
    bank.touch()
    monkeypatch.setattr(banks, "load_bank", Mock(return_value=(None, {"normal_videos": [
        {"name": "Other.2002_label_B1", "path": str(root / "Testing_Video/Other.2002_label_B1.mp4")}]})))
    with pytest.raises(ValueError, match="overlap"):
        cli.main("XD-Violence", args + ["--stage", "evaluate", "--frozen-reference-memory",
                                        "--memory-bank-path", str(bank)])
    workers.assert_not_called()


def test_xd_frozen_memory_requires_evaluate(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    with pytest.raises(SystemExit):
        cli.main("XD-Violence", args + ["--frozen-reference-memory"])
    workers.assert_not_called()


def test_visual_xd_defaults_use_xd_bank_not_ucf_config_bank(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    path = Path(args[1])
    config = yaml.safe_load(path.read_text())
    config["memory"] = {"representation": "visual_text"}
    config["evaluation"].update(memory_bank_path="outputs/memory/ucf_reference.npz",
                                checkpoint_dir="outputs/checkpoints/ucf", output_dir="outputs/ucf")
    path.write_text(yaml.safe_dump(config))
    cli.main("XD-Violence", args)
    assert finalize.call_args.args[6].name == "xd_violence_visual_text_v1.npz"
    assert workers.call_args.args[2].name == "xd_violence_multi_gpu"
    assert not finalize.call_args.args[1]["evaluation"].get("frozen_reference_memory")


def test_ucf_cli_honors_configured_memory_and_checkpoints(xd_cli):
    cli, root, args, workers, finalize = xd_cli
    path = Path(args[1])
    annotation = root / "ucf_annotations.txt"
    annotation.write_text("Other.2002_label_A.mp4 Normal -1 -1\nOther.2002_label_B1.mp4 Fighting 1 3\n")
    config = yaml.safe_load(path.read_text())
    config["evaluation"].update(normal_dir=str(root / "Normal_Videos"), test_dir=str(root / "Testing_Video"),
        annotation_file=str(annotation), memory_bank_path="outputs/memory/selected_visual_bank.npz",
        checkpoint_dir="outputs/checkpoints/selected", output_dir="outputs/selected_evaluation")
    path.write_text(yaml.safe_dump(config))
    cli.main("UCF-Crime", ["--config", str(path), "--gpus", "0,1"])
    assert finalize.call_args.args[6].name == "selected_visual_bank.npz"
    assert workers.call_args.args[2].name == "selected"
    assert finalize.call_args.args[8].parent.name == "selected_evaluation"
