"""Config layering, the memory architecture (B, B+J) and observe-then-detect, without models."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from anomaly_detection.config import load_config
from anomaly_detection.datatypes import EventChunk, MemoryEntry
from anomaly_detection.memory.visual_text import VisualTextMemoryBank
from anomaly_detection.scoring import ARCHITECTURES, architecture, build_scorer, needs_bank, uses_judge
from anomaly_detection.scoring.judge import (MemoryJudge, RatingCache, expected_digit, format_prompt,
                                             retrieve)
from anomaly_detection.scoring.memory_novelty import (MemoryNoveltyScorer, NoveltyMemory, center_rows,
                                                      max_similarity, robust_z, robust_z_params)
from anomaly_detection.temporal.projection import project_window_scores, smoothing_width

ROOT = Path(__file__).resolve().parents[1]


def unit(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return (rows / np.linalg.norm(rows, axis=-1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- configs

def test_extends_and_overlays_deep_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"a": {"x": 1, "y": [1, 2]}, "b": 1}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"extends": ["base.yaml"], "a": {"y": [3]}, "c": 2}))
    (tmp_path / "overlay.yaml").write_text(yaml.safe_dump({"a": {"x": 5}}))
    assert load_config(tmp_path / "child.yaml", [tmp_path / "overlay.yaml"]) == \
        {"a": {"x": 5, "y": [3]}, "b": 1, "c": 2}
    (tmp_path / "loop.yaml").write_text(yaml.safe_dump({"extends": "loop.yaml"}))
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "loop.yaml")


@pytest.mark.parametrize("path", sorted((ROOT / "config/architectures").glob("*.yaml")))
def test_every_architecture_config_loads(path):
    config = load_config(path, [ROOT / "config/datasets/ucf.yaml"])
    assert architecture(config) in ARCHITECTURES
    assert config["temporal"]["mode"] == "sliding_windows"
    assert "masking" not in config


def test_architecture_flags():
    b = load_config(ROOT / "config/architectures/memory_b.yaml")
    bj = load_config(ROOT / "config/architectures/b_plus_j.yaml")
    sb = load_config(ROOT / "config/architectures/shared_best.yaml")
    od = load_config(ROOT / "config/architectures/observe_detect.yaml")
    assert needs_bank(b) and needs_bank(bj) and not needs_bank(sb) and not needs_bank(od)
    assert not uses_judge(b) and uses_judge(bj)
    assert bj["memory_novelty"]["image_weight"] == .25 and bj["memory_novelty"]["judge_weight"] == 1
    assert bj["temporal"]["smoothing_seconds"] == 36 and bj["temporal"]["smoothing_divisor"] == 4
    with pytest.raises(ValueError, match="observe_detect"):
        build_scorer(od, encoder=None)
    with pytest.raises(ValueError, match="architecture"):
        architecture({"architecture": "memory"})


# ---------------------------------------------------------------- temporal

@pytest.mark.parametrize("frames,fps,expected_seconds", [(30 * 400, 30., 36.), (30 * 40, 30., 10.), (25 * 8, 25., 2.)])
def test_length_adaptive_smoothing(frames, fps, expected_seconds):
    width = smoothing_width({"smoothing_seconds": 36, "smoothing_divisor": 4}, frames, fps)
    expected = max(1, int(round(expected_seconds * fps)))
    assert width == expected + (expected % 2 == 0)
    assert smoothing_width({"smoothing_seconds": 0}, frames, fps) == 0


def test_unclipped_projection_keeps_z_scores():
    events = [SimpleNamespace(start_frame=0, stop_frame=4), SimpleNamespace(start_frame=2, stop_frame=6)]
    scores = project_window_scores(events, [-3., 5.], 6, 1., {}, clip=False)
    np.testing.assert_allclose(scores, [-3, -3, 1, 1, 5, 5])
    assert project_window_scores(events, [-3., 5.], 6, 1., {}).min() == 0


# ---------------------------------------------------------------- memory novelty (B)

def test_robust_z_matches_research_formula():
    reference = np.r_[np.arange(20.), np.nan]
    params = robust_z_params(reference)
    filled = np.where(np.isfinite(reference), reference, np.nanmedian(reference))
    q1, q3 = np.percentile(filled, [25, 75])
    assert params == {"median": float(np.nanmedian(reference)), "scale": float((q3 - q1) / 1.349)}
    np.testing.assert_allclose(robust_z([params["median"], np.nan], params), [0, 0])
    constant = robust_z_params([1., 1., 1., 1., 1., 5.])  # IQR 0 -> std
    assert constant["scale"] == pytest.approx(np.std([1., 1., 1., 1., 1., 5.]))


def test_centering_and_leave_own_video_out():
    x = unit([[1, 0, 0], [1, .2, 0], [0, 1, 0], [0, 1, .3]])
    groups = np.array([0, 0, 1, 1])
    centered = center_rows(x, groups, 1.0)
    for g in (0, 1):
        rows = x[groups == g].astype(np.float64)
        residual = rows - rows.mean(0)
        np.testing.assert_allclose(centered[groups == g],
                                   residual / np.linalg.norm(residual, axis=1, keepdims=True), atol=1e-6)
    sims = max_similarity(x, x, groups, groups)
    expected = [max(x[i] @ x[j] for j in range(4) if groups[j] != groups[i]) for i in range(4)]
    np.testing.assert_allclose(sims, expected, atol=1e-6)


class TextEncoder:
    name = "fake-text"

    def __init__(self, table):
        self.table = table

    def encode(self, captions):
        return np.stack([self.table[c] for c in captions])


def novelty_fixture(rng=None):
    rng = rng or np.random.default_rng(3)
    captions = [f"normal {i}" for i in range(6)] + ["fight", "fire"]
    table = {c: unit(rng.normal(size=8)) for c in captions}
    # Memory: 3 videos x 4 windows; captions repeat within and across videos.
    window_captions = ["normal 0", "normal 1", "normal 2", "normal 0",
                       "normal 3", "normal 1", "normal 4", "normal 3",
                       "normal 5", "normal 2", "normal 5", "normal 0"]
    entries = list(dict.fromkeys(window_captions))
    bank = VisualTextMemoryBank([MemoryEntry(t, table[t]) for t in entries],
                                unit(rng.normal(size=(12, 5))), [entries.index(c) for c in window_captions],
                                ["v0"] * 4 + ["v1"] * 4 + ["v2"] * 4, list(range(12)),
                                {"visual": {"center": .2, "scale": .1}, "text": {"center": .2, "scale": .1}})
    return bank, TextEncoder(table), window_captions


def reference_b(bank, encoder, window_captions, test_captions, test_images, alpha=1.0, w=.25):
    """Independent re-implementation of the research B (memory_terms in hazard_scores.py)."""
    def center(x, video):
        out = np.empty_like(x, dtype=np.float64)
        for v in np.unique(video):
            e = x[video == v].astype(np.float64)
            r = e - alpha * e.mean(0)
            out[video == v] = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
        return out

    def rz(t, b):
        med = np.median(b)
        q1, q3 = np.percentile(b, [25, 75])
        return (t - med) / ((q3 - q1) / 1.349 or np.std(b) or 1.0)
    _, bvideo = np.unique(bank.visual_sources, return_inverse=True)
    xb = center(encoder.encode(window_captions), bvideo)
    xt = center(encoder.encode(test_captions), np.zeros(len(test_captions), int))
    same = bvideo[:, None] == bvideo[None, :]
    bt = 1 - np.where(same, -4, xb @ xb.T).max(1)
    bi = 1 - np.where(same, -4, bank.visual_embeddings @ bank.visual_embeddings.T).max(1)
    tt, ti = 1 - (xt @ xb.T).max(1), 1 - (test_images @ bank.visual_embeddings.T).max(1)
    return rz(tt, bt) + w * rz(ti, bi)


def test_memory_novelty_b_matches_independent_reference(tmp_path):
    bank, encoder, window_captions = novelty_fixture()
    cache = tmp_path / "bank.novelty.json"
    memory = NoveltyMemory(bank, encoder, alpha=1.0, cache_path=cache)
    scorer = MemoryNoveltyScorer({"image_weight": .25}, memory, encoder)
    test_captions = ["normal 1", "fight", "fire", "normal 4"]
    images = unit(np.random.default_rng(9).normal(size=(4, 5)))
    events = [EventChunk(i, i, i, i + 4, description=c, average_embedding=images[i], stop_frame=i + 4)
              for i, c in enumerate(test_captions)]
    results = scorer.score_video(events)
    expected = reference_b(bank, encoder, window_captions, test_captions, images)
    np.testing.assert_allclose([r.anomaly_score for r in results], expected, atol=1e-5)
    assert all(r.component_scores["score"] == r.anomaly_score for r in results)
    # The cached reference statistics are reused, and refused for another bank.
    assert json.loads(cache.read_text())["key"]["alpha"] == 1.0
    again = NoveltyMemory(bank, encoder, alpha=1.0, cache_path=cache)
    assert again.text_params == memory.text_params
    with pytest.raises(ValueError, match="different bank"):
        NoveltyMemory(bank, encoder, alpha=1.25, cache_path=cache)


def test_unit_scale_is_monotonic_with_threshold_at_half():
    bank, encoder, _ = novelty_fixture()
    scorer = MemoryNoveltyScorer({"threshold_z": 2.0}, NoveltyMemory(bank, encoder), encoder)
    values = scorer.unit([-10, 0, 2, 3, 50])
    assert values[2] == .5 and np.all(np.diff(values) > 0) and values.min() >= 0 and values.max() <= 1
    with pytest.raises(ValueError, match="judge"):
        MemoryNoveltyScorer({"judge_weight": 1.0}, NoveltyMemory(bank, encoder), encoder)


# ---------------------------------------------------------------- judge (J)

def test_expected_digit_from_logprobs_and_text_fallback():
    response = {"logprobs": [{"top_logprobs": [{"token": "7", "logprob": np.log(.5)},
                                               {"token": " 8", "logprob": np.log(.3)},
                                               {"token": "x", "logprob": np.log(.1)},
                                               {"token": "2", "logprob": np.log(.2)}]}]}
    assert expected_digit(response) == pytest.approx((7 * .5 + 8 * .3 + 2 * .2) / 1.0)
    assert expected_digit({"message": {"content": "4"}}) == 4.0
    with pytest.raises(RuntimeError):
        expected_digit({"message": {"content": "maybe"}})


def test_retrieve_orders_dedupes_and_excludes_own_video_only_captions():
    memory = unit([[1, 0], [.9, .1], [.8, .2], [0, 1]])
    texts = ["a", "b", "a", "c"]
    assert retrieve(unit([[1, 0]]), memory, texts, 2) == [["a", "b"]]
    only_group = np.array([0, -1, -1, 1])
    assert retrieve(unit([[1, 0]]), memory, texts, 3, query_groups=[0], entry_only_group=only_group) == [["b", "a", "c"]]
    assert retrieve(unit([[1, 0]]), memory, texts, 3, query_groups=[1], entry_only_group=only_group) == [["a", "b"]]


class FakeRater:
    def __init__(self):
        self.calls = []

    def rate(self, prompts, on_result=None):
        self.calls.append(list(prompts))
        out = {}
        for p in prompts:
            out[p] = 9.0 if "New moment: fight" in p else 0.5 + 0.1 * (len(p) % 5)
            if on_result:
                on_result(p, out[p])
        return out


def test_judge_calibrates_on_memory_caches_and_scores(tmp_path):
    bank, encoder, window_captions = novelty_fixture()
    memory = NoveltyMemory(bank, encoder)
    rater, cache = FakeRater(), RatingCache(tmp_path / "ratings.json")
    judge = MemoryJudge({"memories": 2, "calibration_windows": 5, "seed": 0}, memory, encoder, rater, cache)
    raw, z = judge.score(["fight", "normal 2", "fight"])
    assert raw[0] == raw[2] == 9.0 and z[0] > 3
    calibration = judge.calibration
    assert calibration["windows"] == 5
    # The calibration prompts are memory windows whose retrieved memories avoid own-video-only captions.
    rows = np.sort(np.random.default_rng(0).choice(12, 5, replace=False))
    prompts = rater.calls[0]
    assert len(prompts) == len(set(prompts)) <= 5
    for row in rows:
        assert any(p.startswith(format_prompt(window_captions[row], [])[:60]) for p in prompts)
    # Re-scoring needs no new ratings; the cache file is reused by a fresh judge.
    judge.score(["fight"])
    fresh = MemoryJudge({"memories": 2, "calibration_windows": 5}, memory, encoder, None,
                        RatingCache(tmp_path / "ratings.json"))
    np.testing.assert_allclose(fresh.score(["fight", "normal 2", "fight"])[1], z)
    with pytest.raises(RuntimeError, match="not cached"):
        fresh.score(["fire"])


def test_b_plus_j_adds_weighted_judge(tmp_path):
    bank, encoder, _ = novelty_fixture()
    memory = NoveltyMemory(bank, encoder)
    judge = MemoryJudge({"memories": 2, "calibration_windows": 4}, memory, encoder, FakeRater(),
                        RatingCache(None))
    b = MemoryNoveltyScorer({}, memory, encoder)
    bj = MemoryNoveltyScorer({"judge_weight": 2.0}, memory, encoder, judge)
    images = unit(np.random.default_rng(1).normal(size=(2, 5)))
    events = [EventChunk(i, i, i, i + 4, description=c, average_embedding=images[i], stop_frame=i + 4)
              for i, c in enumerate(["fight", "normal 3"])]
    base = [r.anomaly_score for r in b.score_video(events)]
    combined = bj.score_video(events)
    z = [r.component_scores["z_judge"] for r in combined]
    np.testing.assert_allclose([r.anomaly_score for r in combined], np.add(base, np.multiply(2.0, z)))


# ---------------------------------------------------------------- end to end

def test_runner_scores_memory_architecture(tmp_path, monkeypatch):
    from anomaly_detection.evaluation.dataset import Annotation
    from anomaly_detection.evaluation import runner
    bank, encoder, _ = novelty_fixture()
    config = {"architecture": "memory_novelty", "memory_novelty": {"image_weight": .25},
              "temporal": {"mode": "sliding_windows", "smoothing_seconds": 36, "smoothing_divisor": 4},
              "evaluation": {"llm_review": False, "annotation_fps": "native", "dataset": "test"}}
    rng = np.random.default_rng(4)

    def video(name):
        captions = ["normal 0", "fight", "fire", "normal 1"] if "anomaly" in name else ["normal 2", "normal 3"]
        events = [EventChunk(4 * i, 4 * i + 3, 4. * i, 4. * i + 4, description=c,
                             average_embedding=unit(rng.normal(size=5)), stop_frame=4 * i + 4)
                  for i, c in enumerate(captions)]
        return SimpleNamespace(events=events, frame_count=4 * len(captions), fps=1.0)
    processor = SimpleNamespace(process=video, encoder=None)
    testing = {"anomaly_a": tmp_path / "anomaly_a.mp4", "normal_a": tmp_path / "normal_a.mp4"}
    annotations = {"anomaly_a": Annotation("anomaly_a", "Fighting", ((5, 12),)),
                   "normal_a": Annotation("normal_a", "Normal", ())}
    for p in testing.values():
        p.touch()
    (tmp_path / "ann.txt").write_text("x")
    monkeypatch.setattr(runner, "save_curves", lambda *a: None)
    scorer = MemoryNoveltyScorer(config["memory_novelty"], NoveltyMemory(bank, encoder), encoder)
    summary = runner.evaluate(processor, bank, {"normal_video_count": 3}, testing, annotations, config,
                              None, tmp_path / "ann.txt", tmp_path / "out", scorer=scorer)
    assert summary["architecture"] == "memory_novelty"
    assert 0 <= summary["overall"]["roc_auc"] <= 1
    with np.load(tmp_path / "out/predictions/anomaly_a.npz") as stored:
        assert stored["scores"].min() >= 0 and stored["scores"].max() <= 1


def test_observe_detect_uses_only_its_own_beginning():
    from anomaly_detection.scoring.observe_detect import run_observe_detect
    from anomaly_detection.perception.windows import build_sliding_windows
    from anomaly_detection.datatypes import EncodedFrame, Frame
    normal, odd = unit([1, 0]), unit([0, 1])
    frames = [Frame(i, float(i), None) for i in range(20)]
    encoded = [EncodedFrame(f, normal if f.frame_number < 14 else odd) for f in frames]
    windows = build_sliding_windows(encoded, 20, 1.0, 4.0, 1.0)

    class Processor:
        encoder = SimpleNamespace(encode_text=lambda text: normal if text == "walking" else odd)

        def caption_events(self, events, raw_frames):
            for e in events:
                e.description = "walking" if e.stop_frame <= 14 else "fighting"
    video = SimpleNamespace(events=windows, frames=frames, frame_count=20)
    reviewed = []
    constitution, memory, results = run_observe_detect(
        video, Processor(), {"observation_fraction": .4, "threshold": .5, "top_k": 10},
        review=lambda result, notes: reviewed.append(notes) or result)
    assert constitution == ["walking"] and len(memory) == 1
    assert all(r.event.start_frame >= 8 for r in results)
    assert not results[0].is_anomaly and results[-1].is_anomaly
    assert reviewed and reviewed[0] == ["walking"]


def test_gpu_float16_retrieval_matches_cpu_on_separated_memories():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(5)
    memory = unit(rng.normal(size=(40, 16)))
    texts = [f"m{i % 30}" for i in range(40)]
    queries = unit(memory[[0, 7, 33]] + .05 * rng.normal(size=(3, 16)))
    only = np.where(np.arange(40) % 2 == 0, 0, -1)
    for groups in (None, [0, 1, 0]):
        kwargs = {} if groups is None else {"query_groups": groups, "entry_only_group": only}
        assert retrieve(queries, memory, texts, 3, device="cuda:0", **kwargs) == \
            retrieve(queries, memory, texts, 3, **kwargs)
