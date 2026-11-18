"""Action scoring and visual normal-memory regressions without model downloads."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anomaly_detection.memory.store import build_bank, load_bank
from anomaly_detection.scoring.action import ActionMemoryScorer, ActionScorer
from anomaly_detection.memory.visual_text import NeighborIndex, calibrate, fit_calibration
from anomaly_detection.datatypes import EventChunk


def unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def test_action_image_weight_endpoints_and_legacy_default():
    encoder = SimpleNamespace(encode_text=lambda t: np.array([1., 0.]) if t == "normal" else np.array([0., 1.]))
    event = EventChunk(0, 1, 0, 1, description="normal", average_embedding=np.array([0., 1.]))
    base = {"normal_prompts": ["normal"], "anomalous_prompts": ["fight"]}
    half = ActionScorer(encoder, base).score(event)
    assert half["action_score"] == pytest.approx(.5)
    for weight, component in ((0, "action_text_score"), (1, "action_image_score")):
        result = ActionScorer(encoder, {**base, "image_weight": weight}).score(event)
        assert result["action_score"] == result[component]
    for invalid in (float("nan"), -1, 2):
        with pytest.raises(ValueError, match="image_weight"):
            ActionScorer(encoder, {**base, "image_weight": invalid})


def test_cached_action_batch_matches_event_scoring():
    vectors = {"walking": unit([1, .2, .1]), "talking": unit([1, .1, .3]),
               "fighting": unit([.1, 1, .2]), "collision": unit([.2, .1, 1])}
    encoder = SimpleNamespace(encode_text=vectors.__getitem__)
    scorer = ActionScorer(encoder, {"normal_prompts": ["walking", "talking"],
        "anomalous_prompts": ["fighting", "collision"], "image_weight": .875})
    images = unit([[.1, .8, .3], [.9, .2, .2]])
    texts = np.stack([vectors["walking"], vectors["collision"]])
    batch = scorer.score_vectors(images, texts)
    for i, caption in enumerate(("walking", "collision")):
        event = EventChunk(0, 1, 0, 1, description=caption, average_embedding=images[i])
        scalar = scorer.score(event)
        for key in scalar:
            assert batch[key][i] == pytest.approx(scalar[key], abs=1e-6)
    missing = scorer.score_vectors(images)
    np.testing.assert_allclose(missing["action_text_score"], missing["action_image_score"])
    for invalid in ("walking", ["walking", "walking"], [""]):
        with pytest.raises(ValueError):
            ActionScorer(encoder, {"normal_prompts": invalid, "anomalous_prompts": ["fighting"]})


@pytest.mark.parametrize("reference_batch", [1, 3, 10])
def test_chunked_neighbors_equal_full_matrix(reference_batch):
    rng = np.random.default_rng(7)
    reference, queries = unit(rng.normal(size=(9, 5))), unit(rng.normal(size=(4, 5)))
    index = NeighborIndex(reference, query_batch=2, reference_batch=reference_batch)
    for k in (1, 3, 20):
        sims = queries @ reference.T
        top = np.sort(sims, axis=1)[:, -min(k, len(reference)):]
        expected = 1 - (.6 * top.mean(1) + .4 * top.max(1))
        np.testing.assert_allclose(index.distances(queries, k), expected, atol=2e-7)


def test_cuda_neighbors_match_cpu_when_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(12)
    reference, queries = unit(rng.normal(size=(19, 7))), unit(rng.normal(size=(5, 7)))
    cpu = NeighborIndex(reference, query_batch=2, reference_batch=6)
    gpu = NeighborIndex(reference, "cuda:0", query_batch=2, reference_batch=6)
    np.testing.assert_allclose(gpu.distances(queries, 4), cpu.distances(queries, 4), atol=2e-7)
    assert gpu.closest(queries[0]) == cpu.closest(queries[0])


def test_calibration_finite_for_constant_normals():
    parameters = fit_calibration([.2] * 10)
    scores = calibrate([.1, .2, .3], parameters)
    assert np.isfinite(scores).all() and scores[0] < scores[1] < scores[2]
    assert scores[1] == .5


def test_visual_bank_persistence_and_scalar_scoring(tmp_path):
    config = {"encoder": {}, "video": {"target_fps": 1}, "segmentation": {},
              "memory_bank": {"min_caption_occurrences": 1}}
    normals = {}
    for i in range(6):
        p = tmp_path / f"normal_{i}.mp4"; p.touch(); normals[p.stem] = p
    vectors = unit([[1, .1], [.1, 1]])
    encoder = SimpleNamespace(encode_text=lambda text: vectors[int(text)])
    def process(path):
        i = int(Path(path).stem.split("_")[-1])
        return SimpleNamespace(events=[EventChunk(0, 1, 0, 1, description=str(i % 2),
                               average_embedding=vectors[i % 2], stop_frame=2)])
    bank, metadata = build_bank(SimpleNamespace(process=process, encoder=encoder), normals,
                                config, tmp_path / "bank.npz")
    reference = {v["name"] for v in metadata["reference_videos"]}
    calibration = {v["name"] for v in metadata["calibration_videos"]}
    assert not reference & calibration
    assert set(bank.visual_sources) == reference
    restored, _ = load_bank(tmp_path / "bank.npz", config, normals)
    np.testing.assert_array_equal(restored.visual_embeddings, bank.visual_embeddings)
    event = process(normals["normal_0"]).events[0]
    settings = {"memory_weight": 1.0, "memory_image_weight": 1, "threshold": .5,
                "normal_prompts": ["0"], "anomalous_prompts": ["1"]}
    result = ActionMemoryScorer(settings, encoder, restored).score_event(event)
    assert result.anomaly_score == result.component_scores["memory_image_score"]
    with pytest.raises(ValueError, match="settings differ"):
        load_bank(tmp_path / "bank.npz", {**config, "memory_bank": {"top_k": 3}})
    with pytest.raises(ValueError, match="features"):
        load_bank(tmp_path / "bank.npz", {**config, "video": {"target_fps": 2}})
    with pytest.raises(FileExistsError):
        build_bank(SimpleNamespace(process=process, encoder=encoder), normals, config, tmp_path / "bank.npz")
