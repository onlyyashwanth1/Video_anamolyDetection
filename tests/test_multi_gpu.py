"""CPU-only regressions; never load pretrained models or touch CUDA."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from anomaly_detection.evaluation.multi_gpu import (
    cache_lock, pending_jobs, resolve_devices, run_workers,
)
from anomaly_detection.evaluation.video_cache import VideoCache, CachedProcessor
from anomaly_detection.utils.types import EventChunk


@pytest.fixture
def config():
    return {
        "encoder": {"model_name": "fake"},
        "video": {"target_fps": 1},
        "segmentation": {"coarse_threshold": .9, "fine_threshold": .95},
        "temporal": {"mode": "sliding_windows", "window_seconds": 4, "stride_seconds": 1},
        "captioning": {"mode": "frame_captions_then_summary", "max_frames_per_window": 4},
        "familiarisation": {"min_caption_occurrences": 1},
        "inference": {"anomaly_threshold": .5},
        "evaluation": {"llm_review": False, "annotation_fps": "native"},
    }


def video(description="walking"):
    return SimpleNamespace(frame_count=40, fps=10., events=[
        EventChunk(0, 30, 0., 4., frame_indices=[0, 1, 2, 3],
                   description=description, average_embedding=np.array([1., 0.]), stop_frame=40)])


def test_cache_roundtrip_and_invalidation(tmp_path, config):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"video")
    cache = VideoCache(tmp_path / "cache", config)
    assert cache.load(source) is None
    cache.save(source, video())
    loaded = cache.load(source)
    assert loaded.frame_count == 40 and loaded.fps == 10
    assert loaded.events[0].description == "walking"
    assert loaded.events[0].stop_frame == 40
    np.testing.assert_array_equal(loaded.events[0].average_embedding, [1, 0])
    # Scoring-only changes can reuse captions, feature changes cannot.
    other = copy.deepcopy(config)
    other["inference"]["anomaly_threshold"] = .8
    assert VideoCache(cache.folder, other).load(source) is not None
    other["temporal"]["stride_seconds"] = 2
    assert VideoCache(cache.folder, other).load(source) is None
    source.write_bytes(b"changed video")
    assert cache.load(source) is None


def test_cache_rejects_corruption_and_changed_source(tmp_path, config):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"video")
    cache = VideoCache(tmp_path / "cache", config)
    identity = cache.identity(source)
    source.write_bytes(b"longer video")
    with pytest.raises(ValueError, match="changed during"):
        cache.save(source, video(), identity)
    cache.save(source, video())
    cache.path(source).write_bytes(b"partial write")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        cache.load(source)


def test_cache_ignores_unpublished_temporary_file(tmp_path, config):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"video")
    cache = VideoCache(tmp_path, config)
    cache.path(source).with_suffix(".interrupted.tmp").write_bytes(b"partial")
    assert cache.load(source) is None


def test_gpu_ids_respect_allocation():
    assert resolve_devices("0,2") == ["0", "2"]
    assert resolve_devices("1,0", "GPU-aaa,GPU-bbb") == ["GPU-bbb", "GPU-aaa"]
    assert resolve_devices("0,1", "3,5") == ["3", "5"]
    for selection, allocation in [("0,0", None), ("0,00", None), ("1", "3"),
                                   ("0", ""), ("0", "-1"), ("-1", None)]:
        with pytest.raises(ValueError):
            resolve_devices(selection, allocation)


def test_longest_first_and_skip_completed(tmp_path, config, monkeypatch):
    import cv2
    paths = {}
    for name in ("short", "long", "cached"):
        paths[name] = tmp_path / f"{name}.mp4"
        paths[name].write_bytes(b"video")
    cache = VideoCache(tmp_path / "cache", config)
    cache.save(paths["cached"], video())

    class Capture:
        def __init__(self, path):
            assert Path(path).stem != "cached"
            self.frames = 300 if Path(path).stem == "long" else 30

        def get(self, key):
            return 30 if key == cv2.CAP_PROP_FPS else self.frames

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    jobs, resumed = pending_jobs(paths, cache)
    assert resumed == 1
    assert [j["name"] for j in jobs] == ["long", "short"]
    assert [j["seconds"] for j in jobs] == [10, 1]


def fake_worker(device, config, folder, threads, jobs, messages):
    """Spawn-safe test worker exercises actual IPC without importing models."""
    while True:
        job = jobs.get()
        if job is None:
            return
        messages.put(("started", device, job["name"]))
        # Exclusive create catches a job accidentally processed twice.
        with (Path(folder) / job["name"]).open("x") as handle:
            handle.write(device)
        time.sleep(.03)
        messages.put(("done", device, job["name"], .03))


def crashed_worker(device, config, folder, threads, jobs, messages):
    os._exit(7)


def test_spawned_workers_complete_each_job_once(tmp_path):
    jobs = [{"name": str(i)} for i in range(8)]
    run_workers(["0", "1"], {}, tmp_path, jobs, worker_target=fake_worker)
    assert {p.name for p in tmp_path.iterdir()} == {str(i) for i in range(8)}
    assert {p.read_text() for p in tmp_path.iterdir()} <= {"0", "1"}


def test_abrupt_worker_failure_does_not_hang(tmp_path):
    with pytest.raises(RuntimeError, match="exited"):
        run_workers(["0"], {}, tmp_path, [{"name": "a"}], worker_target=crashed_worker)


def test_lock_prevents_concurrent_schedulers(tmp_path):
    with cache_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another evaluator"):
            with cache_lock(tmp_path):
                pass
    with cache_lock(tmp_path):
        pass


def test_cached_pipeline_matches_serial_memory_and_metrics(tmp_path, config, monkeypatch):
    from anomaly_detection.evaluation.bank import build_bank
    from anomaly_detection.evaluation.dataset import Annotation
    from anomaly_detection.evaluation.runner import evaluate
    import anomaly_detection.evaluation.runner as runner

    class Encoder:
        def encode_text(self, text):
            return np.array([1., 0.]) if text == "walking" else np.array([0., 1.])

    class Processor:
        encoder = Encoder()

        def process(self, source):
            return video("running" if "anomaly" in str(source) else "walking")

    processor = Processor()
    normal, testing = {}, {}
    for group, names in [(normal, ["memory_a", "memory_b"]),
                         (testing, ["normal_test", "anomaly_test"])]:
        for name in names:
            group[name] = tmp_path / f"{name}.mp4"
            group[name].write_bytes(b"video")
    cache = VideoCache(tmp_path / "cache", config)
    for source in [*normal.values(), *testing.values()]:
        cache.save(source, processor.process(source))
    cached = CachedProcessor(cache, Encoder())
    bank1, meta1 = build_bank(processor, normal, config, tmp_path / "one.npz")
    bank2, meta2 = build_bank(cached, normal, config, tmp_path / "two.npz")
    assert [(e.text, e.occurrence_count) for e in bank1.entries] == [
        (e.text, e.occurrence_count) for e in bank2.entries]
    assert meta1["processed_videos"] == meta2["processed_videos"]
    annotations = {"normal_test": Annotation("normal_test", "Normal", ()),
                   "anomaly_test": Annotation("anomaly_test", "Abuse", ((1, 40),))}
    annotation_path = tmp_path / "annotations.txt"
    annotation_path.write_text("test fixture")
    monkeypatch.setattr(runner, "save_curves", lambda *a: None)
    results = []
    for i, (p, bank, meta) in enumerate([(processor, bank1, meta1), (cached, bank2, meta2)]):
        results.append(evaluate(p, bank, meta, testing, annotations, config,
                                tmp_path / ("one.npz" if i == 0 else "two.npz"),
                                annotation_path, tmp_path / f"result{i}"))
    assert results[0] == results[1]
    for name in testing:
        with np.load(tmp_path / "result0/predictions" / f"{name}.npz") as a:
            with np.load(tmp_path / "result1/predictions" / f"{name}.npz") as b:
                np.testing.assert_array_equal(a["scores"], b["scores"])
