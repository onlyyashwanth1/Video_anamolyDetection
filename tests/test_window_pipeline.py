"""Window coverage, temporal caption reuse, and action evidence regressions."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly_detection.utils.types import Frame, EncodedFrame, MemoryEntry
from anomaly_detection.segmentation.sliding_windows import build_sliding_windows, split_familiarisation
from anomaly_detection.evaluation.dataset import expand_windows
from anomaly_detection.familiarisation.domain_familiarisation import caption_windows
from anomaly_detection.inference.action_scoring import ActionScorer
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.evaluation.bank import model_spec, save_bank, load_bank


def samples(count, fps=10):
    return [EncodedFrame(Frame(i, i / fps, np.zeros((2, 2, 3), dtype=np.uint8)),
                         np.array([1., 0.])) for i in range(0, count, fps)]


@pytest.mark.parametrize("count", [1, 25, 40, 51, 99])
def test_window_coverage_short_and_partial_tail(count):
    windows = build_sliding_windows(samples(count), count, 10)
    assert windows[0].start_frame == 0
    assert windows[-1].stop_frame == count
    np.testing.assert_allclose(expand_windows(windows, np.ones(len(windows)), count), 1)
    assert all(w.stop_frame - w.start_frame <= 40 for w in windows)


def test_overlap_averages_and_rejects_gaps():
    windows = build_sliding_windows(samples(50), 50, 10)
    scores = expand_windows(windows, [0.2, 0.8], 50)
    np.testing.assert_allclose(scores[:10], .2)
    np.testing.assert_allclose(scores[10:40], .5)
    np.testing.assert_allclose(scores[40:], .8)
    with pytest.raises(ValueError, match="uncovered"):
        expand_windows(windows[1:], [.8], 50)


def test_observation_split_shares_no_frames():
    windows = build_sliding_windows(samples(200), 200, 10)
    normal, live = split_familiarisation(windows, .4, 200)
    assert max(w.stop_frame for w in normal) <= min(w.start_frame for w in live)
    assert set(i for w in normal for i in w.frame_indices).isdisjoint(
        i for w in live for i in w.frame_indices)


def test_caption_cache_and_order(monkeypatch):
    import anomaly_detection.reasoning as reasoning
    gemma = MagicMock()
    gemma.generate_caption.side_effect = ["walking", "turning", "striking", "falling", "standing"]
    gemma.generate_text.return_value = "A person strikes another."
    monkeypatch.setitem(sys.modules, "anomaly_detection.reasoning.local_gemma", gemma)
    monkeypatch.setattr(reasoning, "local_gemma", gemma, raising=False)
    encoded = samples(50)
    windows = build_sliding_windows(encoded, 50, 10)
    caption_windows(windows, None, [s.frame for s in encoded], strict=True)
    assert gemma.generate_caption.call_count == 5  # not 8 for the overlapping windows
    prompt = gemma.generate_text.call_args_list[0].args[0]
    assert prompt.index("0.000s: walking") < prompt.index("3.000s: falling")
    assert all(w.description == "A person strikes another." for w in windows)
    gemma.generate_text.return_value = ""
    gemma.generate_caption.side_effect = None
    gemma.generate_caption.return_value = "walking"
    with pytest.raises(RuntimeError, match="summarization"):
        caption_windows(windows, None, [s.frame for s in encoded], strict=True)


def test_action_evidence_detects_familiar_anomaly():
    encoder = SimpleNamespace(encode_text=lambda text: np.array([1., 0.])
                              if text == "normal" else np.array([0., 1.]))
    scorer = ActionScorer(encoder, {"normal_prompts": ["normal"], "anomalous_prompts": ["fight"]})
    event = build_sliding_windows(samples(20), 20, 10)[0]
    event.average_embedding = np.array([0., 1.])
    event.description = "fight"
    # Deliberately familiar visual/text evidence: memory alone must miss it.
    bank = MemoryBank([MemoryEntry("fight", np.array([0., 1.]))])
    baseline = score_event(event, bank, .5, encoder)
    result = score_event(event, bank, .5, encoder, action_scorer=scorer,
                         settings={"scoring_mode": "action"})
    assert not baseline.is_anomaly and result.is_anomaly
    assert result.component_scores["action_text_margin"] > 0
    assert result.component_scores["memory_score"] == baseline.anomaly_score
    event.average_embedding = np.array([1., 0.])
    event.description = "normal"
    assert scorer.score(event)["action_score"] < .5


def test_memory_version_rejects_changed_caption_settings(tmp_path):
    config = {"encoder": {}, "video": {"target_fps": 1}, "segmentation": {},
              "familiarisation": {"min_caption_occurrences": 1},
              "captioning": {"mode": "frame_captions_then_summary", "max_frames_per_window": 4}}
    bank = MemoryBank([MemoryEntry("normal", np.array([1., 0.]))])
    path = tmp_path / "bank.npz"
    save_bank(path, bank, {"schema": 1, "model_spec": model_spec(config)})
    load_bank(path, config)
    config["captioning"]["max_frames_per_window"] = 2
    with pytest.raises(ValueError, match="different model"):
        load_bank(path, config)
