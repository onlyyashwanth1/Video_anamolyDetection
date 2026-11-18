"""The default action+memory model reaches evaluation and single-video scoring unchanged."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from anomaly_detection.config import load_config
from anomaly_detection.datatypes import EventChunk, MemoryEntry
from anomaly_detection.memory.visual_text import VisualTextMemoryBank
from anomaly_detection.scoring.action import ActionMemoryScorer

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return load_config(ROOT / "config/architectures/action_default.yaml", [ROOT / "config/datasets/ucf.yaml"])


def test_default_config_identifies_the_thesis_model():
    config = configuration()
    prompts = yaml.safe_load((ROOT / "research/config/ucf_action_prompts_expanded.yaml").read_text())
    assert config["architecture"] == "action"
    action = config["action"]
    assert action["memory_weight"] == .05
    assert action["image_weight"] == .875
    assert action["memory_image_weight"] == 1
    assert action["threshold"] == pytest.approx(0.5935198772337466)
    for key in ("normal_prompts", "anomalous_prompts"):
        assert action[key] == prompts[key]
        assert len(prompts[key]) == 26
    assert not config["evaluation"].get("frozen_reference_memory")
    assert config["reasoning"]["backend"] == "ollama"
    assert "model_digest" not in config["reasoning"]  # resolved from the installed model at run time


@pytest.mark.parametrize("dataset", ["ucf", "xd", "shanghaitech", "avenue"])
def test_dataset_files_change_only_data_paths_and_ports(dataset):
    base = load_config(ROOT / "config/architectures/action_default.yaml")
    merged = load_config(ROOT / "config/architectures/action_default.yaml",
                         [ROOT / "config/datasets" / f"{dataset}.yaml"])
    for key in set(base) | set(merged):
        if key not in ("evaluation", "reasoning"):
            assert merged[key] == base[key]
    assert {k: v for k, v in merged["reasoning"].items() if k != "base_port"} == \
        {k: v for k, v in base["reasoning"].items() if k != "base_port"}


def scoring_objects(config):
    normal, anomalous = np.array([1., 0.], dtype=np.float32), np.array([0., 1.], dtype=np.float32)
    normal_texts = set(config["action"]["normal_prompts"])
    encoder = SimpleNamespace(encode_text=lambda text: normal if text in normal_texts or text == "walking" else anomalous)
    bank = VisualTextMemoryBank([MemoryEntry("walking", normal)], normal[None], [0], ["reference"], [0],
        {"visual": {"center": .2, "scale": .3}, "text": {"center": .2, "scale": .3}})
    event = EventChunk(0, 3, 0, 4, description="walking", average_embedding=anomalous,
                       stop_frame=4, frame_indices=[0, 1, 2, 3])
    return encoder, bank, event


def test_default_scoring_uses_visual_memory_and_requested_blends():
    config = configuration()
    encoder, bank, event = scoring_objects(config)
    result = ActionMemoryScorer(config["action"], encoder, bank).score_event(event)
    c = result.component_scores
    assert c["memory_score"] == c["memory_image_score"]
    assert c["action_score"] == pytest.approx(.875 * c["action_image_score"] + .125 * c["action_text_score"])
    assert result.anomaly_score == pytest.approx(.95 * c["action_score"] + .05 * c["memory_image_score"])
    assert result.closest_match == "walking"


def test_demo_loads_saved_bank_and_scores_entire_video(monkeypatch):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.memory.store as store
    import anomaly_detection.reasoning.ollama_runtime as runtime
    config = deepcopy(configuration())
    config["reasoning"]["backend"] = "transformers"  # fake captions need no Ollama/model startup
    encoder, bank, event = scoring_objects(config)
    processor = Mock()
    processor.encoder = encoder
    processor.process.return_value = SimpleNamespace(events=[event], frames=[], frame_count=4, fps=1.0)
    monkeypatch.setattr(pipeline, "VideoEventProcessor", Mock(return_value=processor))
    loader = Mock(return_value=(bank, {}))
    monkeypatch.setattr(store, "load_bank", loader)
    monkeypatch.setattr(runtime, "caption_session", lambda config: __import__("contextlib").nullcontext())
    observe = Mock(side_effect=AssertionError("Must not rebuild input-video memory"))
    monkeypatch.setattr(pipeline, "run_observe_detect", observe)
    review = Mock(side_effect=AssertionError("Selected model disables verdict review"))
    monkeypatch.setattr(pipeline, "explain_anomaly", review)
    notebook, actual_bank, results = pipeline.run_full_pipeline("demo.mp4", config)
    assert actual_bank is bank and notebook == ["walking"]
    assert len(results) == 1 and results[0].event is event
    loader.assert_called_once_with(ROOT / config["evaluation"]["memory_bank_path"], config)
    processor.caption_events.assert_called_once_with([event], [])
    observe.assert_not_called()
    review.assert_not_called()


def test_missing_bank_fails_before_models_or_servers_start(monkeypatch):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.memory.store as store
    import anomaly_detection.reasoning.ollama_runtime as runtime
    config = configuration()
    config["reasoning"]["backend"] = "transformers"
    monkeypatch.setattr(store, "load_bank", Mock(side_effect=FileNotFoundError("missing bank")))
    session, processor = Mock(), Mock()
    monkeypatch.setattr(runtime, "caption_session", session)
    monkeypatch.setattr(pipeline, "VideoEventProcessor", processor)
    with pytest.raises(FileNotFoundError, match="missing bank"):
        pipeline.run_full_pipeline("demo.mp4", config)
    session.assert_not_called()
    processor.assert_not_called()
