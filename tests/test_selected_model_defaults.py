"""Selected model reaches both evaluation and demo scoring without legacy fallbacks."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from anomaly_detection.inference.action_scoring import ActionScorer
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.memory.visual_memory import VisualTextMemoryBank
from anomaly_detection.utils.types import EventChunk, MemoryEntry

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text())


def test_default_configs_identify_the_requested_model():
    config = configuration()
    ollama = yaml.safe_load((ROOT / "config/ucf_ollama.yaml").read_text())
    prompts = yaml.safe_load((ROOT / "config/ucf_action_prompts_expanded.yaml").read_text())
    assert config == ollama
    assert config["inference"]["scoring_mode"] == "memory_plus_action"
    assert config["inference"]["action_weight"] == .95
    assert config["inference"]["action"]["image_weight"] == .875
    assert config["inference"]["memory_image_weight"] == 1
    assert config["memory"]["representation"] == "visual_text"
    assert config["memory"]["allow_anomaly_promotion"] is False
    for key in ("normal_prompts", "anomalous_prompts"):
        assert config["inference"]["action"][key] == prompts[key]
        assert len(prompts[key]) == 26
    assert not config["evaluation"].get("frozen_reference_memory")
    assert config["reasoning"]["backend"] == "ollama"


def scoring_objects(config):
    normal, anomalous = np.array([1., 0.], dtype=np.float32), np.array([0., 1.], dtype=np.float32)
    normal_texts = set(config["inference"]["action"]["normal_prompts"])
    encoder = SimpleNamespace(encode_text=lambda text: normal if text in normal_texts or text == "walking" else anomalous)
    bank = VisualTextMemoryBank([MemoryEntry("walking", normal)], normal[None], [0], ["reference"], [0],
        {"visual": {"center": .2, "scale": .3}, "text": {"center": .2, "scale": .3}})
    event = EventChunk(0, 3, 0, 4, description="walking", average_embedding=anomalous,
                       stop_frame=4, frame_indices=[0, 1, 2, 3])
    return encoder, bank, event


def test_default_scoring_uses_visual_memory_and_requested_blends():
    config = configuration()
    encoder, bank, event = scoring_objects(config)
    scorer = ActionScorer(encoder, config["inference"]["action"])
    result = score_event(event, bank, config["inference"]["anomaly_threshold"],
                         encoder, action_scorer=scorer, settings=config["inference"])
    c = result.component_scores
    assert c["memory_score"] == c["memory_image_score"]
    assert c["action_score"] == pytest.approx(.875 * c["action_image_score"] + .125 * c["action_text_score"])
    assert result.anomaly_score == pytest.approx(.95 * c["action_score"] + .05 * c["memory_image_score"])


def test_demo_loads_saved_bank_and_scores_entire_video(monkeypatch):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.evaluation.bank as banks
    config = deepcopy(configuration())
    config.pop("reasoning")  # fake captions need no Ollama/model startup
    encoder, bank, event = scoring_objects(config)
    processor = Mock()
    processor.encoder = encoder
    processor.process.return_value = SimpleNamespace(events=[event], frames=[], frame_count=4)
    constructor = Mock(return_value=processor)
    monkeypatch.setattr(pipeline, "VideoEventProcessor", constructor)
    loader = Mock(return_value=(bank, {}))
    monkeypatch.setattr(banks, "load_bank", loader)
    rebuild = Mock(side_effect=AssertionError("Must not rebuild input-video textual memory"))
    monkeypatch.setattr(pipeline, "build_domain_constitution", rebuild)
    review = Mock(side_effect=AssertionError("Selected model disables verdict review"))
    monkeypatch.setattr(pipeline, "explain_anomaly", review)
    notebook, actual_bank, results = pipeline.run_full_pipeline("demo.mp4", config)
    assert actual_bank is bank and notebook == ["walking"]
    assert len(results) == 1 and results[0].event is event
    loader.assert_called_once_with(ROOT / config["evaluation"]["memory_bank_path"], config)
    processor.caption_events.assert_called_once_with([event], [])
    rebuild.assert_not_called()
    review.assert_not_called()


def test_missing_bank_fails_before_models_or_servers_start(monkeypatch):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.evaluation.bank as banks
    import anomaly_detection.reasoning.ollama_runtime as runtime
    config = configuration()
    config.pop("reasoning")
    monkeypatch.setattr(banks, "load_bank", Mock(side_effect=FileNotFoundError("missing bank")))
    session, processor = Mock(), Mock()
    monkeypatch.setattr(runtime, "caption_session", session)
    monkeypatch.setattr(pipeline, "VideoEventProcessor", processor)
    with pytest.raises(FileNotFoundError, match="missing bank"):
        pipeline.run_full_pipeline("demo.mp4", config)
    session.assert_not_called()
    processor.assert_not_called()
