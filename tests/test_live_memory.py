"""Live self-memory mode: split, calibration, scoring and subtitles (no models needed)."""

import numpy as np
import pytest

from anomaly_detection.live.self_memory import (
    SelfMemory, boundary_frame, knn_distance, split_windows, time_guard)
from anomaly_detection.live.subtitles import ANOMALY, NORMAL, OBSERVE, build_cues, build_segments, write_srt
from anomaly_detection.live.judge import retrieve
from anomaly_detection.utils.types import EventChunk

FPS = 10


def windows(seconds, width=4, stride=1):
    frames = seconds * FPS
    out = []
    for start in range(0, frames, stride * FPS):
        stop = min(start + width * FPS, frames)
        out.append(EventChunk(start_frame=start, end_frame=stop - 1, start_time=start / FPS,
                              end_time=stop / FPS, stop_frame=stop, description=f"w{start // FPS}"))
        if stop == frames:
            break
    return out


def test_split_keeps_test_out_of_memory():
    events = windows(100)
    boundary = boundary_frame(100 * FPS, FPS, fraction=.4)
    memory, test = split_windows(events, boundary)
    assert boundary == 400
    assert max(e.stop_frame for e in memory) <= boundary <= min(e.start_frame for e in test)
    assert len(memory) == 37 and test[0].start_time == 40 and test[-1].end_time == 100
    assert boundary_frame(100 * FPS, FPS, seconds=30) == 300
    with pytest.raises(ValueError):
        boundary_frame(1000, FPS, fraction=.4, seconds=3)


def test_time_guard_excludes_self_and_overlaps():
    events = windows(30)
    guard = time_guard(events, 0)
    assert guard[0, 0] and guard[0, 3] and not guard[0, 4]
    assert not time_guard(events, 2 * FPS)[0, 6]


def test_knn_distance_respects_exclusion():
    memory = np.eye(3)
    queries = np.eye(3)
    distance, nearest = knn_distance(queries, memory, 1, np.eye(3, dtype=bool))
    assert np.allclose(distance, 1) and not np.any(nearest == np.arange(3))


def _scene(rng, count, dim=16, offset=None):
    base = np.zeros(dim)
    base[0] = 5
    noise = rng.normal(size=(count, dim)) * .3
    if offset is not None:
        noise[:, offset] += 3
    return base + noise


def test_novel_windows_score_higher_than_normal_ones():
    rng = np.random.default_rng(0)
    memory = windows(60)[:40]
    settings = {"k": 5, "calibration_guard_seconds": 4, "judge_weight": 0}
    model = SelfMemory.build(memory, _scene(rng, 40), _scene(rng, 40), settings, FPS)
    normal = model.score(_scene(rng, 20), _scene(rng, 20))["deviation"]
    novel = model.score(_scene(rng, 20, offset=5), _scene(rng, 20, offset=7))["deviation"]
    assert np.median(normal) < 2 < np.min(novel)
    assert model.calibration["held_out_windows"] == 40
    centered = SelfMemory.build(memory, _scene(rng, 40), _scene(rng, 40), {**settings, "centering": True}, FPS)
    assert np.isfinite(centered.score(_scene(rng, 3), _scene(rng, 3))["deviation"]).all()


def test_judge_component_is_calibrated_and_required():
    rng = np.random.default_rng(1)
    memory = windows(60)[:30]
    settings = {"k": 5, "judge_weight": 1.0}
    model = SelfMemory.build(memory, _scene(rng, 30), _scene(rng, 30), settings, FPS,
                             judge_scores=rng.uniform(0, 2, 30))
    visual, text = _scene(rng, 1), _scene(rng, 1)
    low = model.score(visual, text, np.asarray([1.0]))["deviation"][0]
    high = model.score(visual, text, np.asarray([9.0]))["deviation"][0]
    assert high > low
    with pytest.raises(ValueError):
        model.score(visual, text)


def test_retrieve_returns_distinct_captions():
    memory = np.eye(4)
    picked = retrieve(np.eye(4)[:1], memory, ["a", "a", "b", "c"], 2)
    assert picked == [["a", "b"]]


def test_cues_cover_video_and_start_anomaly_at_onset(tmp_path):
    events = windows(30)
    memory, test = split_windows(events, 12 * FPS)
    deviation = [0.0] * len(test)
    for j, window in enumerate(test):
        if 20 <= window.start_time < 23:
            deviation[j] = 9.0
    segments = build_segments(memory, test, deviation, 30)
    assert segments[0][0] == 0 and segments[-1][1] == 30
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    captions = {("memory", i): w.description for i, w in enumerate(memory)}
    captions.update({("test", j): w.description for j, w in enumerate(test)})
    cues = build_cues(segments, captions, threshold=3.0, cue_seconds=4)
    labels = [c.label for c in cues]
    assert labels[0] == OBSERVE and ANOMALY in labels and NORMAL in labels
    assert all(c.end - c.start <= 4 + 1e-6 for c in cues)
    first = next(c for c in cues if c.label == ANOMALY)
    # Mean pooling over covering windows: onset when most covering windows are anomalous.
    assert 19 <= first.start <= 21 and first.caption == f"w{int(first.start)}"
    path = tmp_path / "out.srt"
    write_srt(cues, path, 3.0, 12.0)
    text = path.read_text()
    assert text.startswith("1\n00:00:00,000 --> ") and "[!! ANOMALY !!]" in text
    assert "<font color=" in text and text.count(" --> ") == len(cues)
    write_srt(cues, path, 3.0, 12.0, colors=False)
    assert "<font" not in path.read_text()


def test_median_fusion_needs_two_components():
    rng = np.random.default_rng(2)
    memory = windows(60)[:30]
    settings = {"k": 5, "judge_weight": 1.0, "fusion": "median"}
    model = SelfMemory.build(memory, _scene(rng, 30), _scene(rng, 30), settings, FPS,
                             judge_scores=rng.uniform(0, 2, 30))
    visual_only = model.score(_scene(rng, 1, offset=5), _scene(rng, 1), np.asarray([1.0]))["deviation"][0]
    both = model.score(_scene(rng, 1, offset=5), _scene(rng, 1), np.asarray([9.0]))["deviation"][0]
    assert visual_only < 3 < both
