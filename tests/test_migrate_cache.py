"""Old masked checkpoints + saved unmasked embeddings -> new checkpoints, captions untouched."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def migrate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("migrate_unmasked_cache")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "config/datasets").mkdir(parents=True)
    base = {"video": {"target_fps": 1.0}, "encoder": {"model_name": "fake"},
            "temporal": {"mode": "sliding_windows", "window_seconds": 4.0, "stride_seconds": 1.0},
            "captioning": {"mode": "frame_captions_then_summary", "max_frames_per_window": 4},
            "segmentation": {}, "reasoning": {"backend": "transformers"}}
    (tmp_path / "config/base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "config/datasets/avenue.yaml").write_text(yaml.safe_dump(
        {"evaluation": {"checkpoint_dir": "new_cache"}}))
    return module, tmp_path, base


def old_checkpoint(folder, source, base, starts, stops, mask="yolov8s-worldv2.pt/bytetrack/blur51"):
    from anomaly_detection.memory.store import feature_spec
    spec = {k: v for k, v in feature_spec(base).items() if k not in ("implementation", "image_preprocessing")}
    spec.update(implementation="window_temporal_caption_v2", masking=mask, min_caption_occurrences=1,
                temporal={"mode": "sliding_windows", "window_seconds": 4.0, "stride_seconds": 1.0})
    stat = source.stat()
    events = [{"start_frame": int(a), "end_frame": int(b) - 1, "start_time": float(a), "end_time": float(b),
               "frame_indices": [int(a)], "description": f"caption {i}", "stop_frame": int(b)}
              for i, (a, b) in enumerate(zip(starts, stops))]
    metadata = {"identity": {"version": 1, "model": spec, "path": str(source.resolve()), "bytes": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}, "events": events, "frame_count": int(stops[-1]), "fps": 1.0}
    folder.mkdir(parents=True, exist_ok=True)
    masked = np.tile([1., 0.], (len(starts), 1))
    np.savez(folder / f"{source.stem}.npz", metadata=json.dumps(metadata), embeddings=masked)


def unmasked(folder, name, starts, stops, vectors):
    (folder / "test").mkdir(parents=True, exist_ok=True)
    np.savez(folder / "test" / f"{name}.npz", embeddings=vectors, start_frame=starts, stop_frame=stops)


def test_migration_keeps_captions_and_swaps_embeddings(migrate, tmp_path):
    module, root, base = migrate
    from anomaly_detection.config import load_config
    from anomaly_detection.evaluation.video_cache import VideoCache
    starts, stops = np.array([0, 1, 2]), np.array([4, 5, 6])
    vectors = np.array([[0., 1.], [.6, .8], [.8, .6]], dtype=np.float32)
    for name in ("good", "shifted", "missing"):
        video = root / f"{name}.mp4"
        video.write_bytes(name.encode())
        old_checkpoint(root / "old", video, base, starts, stops)
    unmasked(root / "unmasked/avenue", "good", starts, stops, vectors)
    unmasked(root / "unmasked/avenue", "shifted", starts + 1, stops + 1, vectors)
    args = ["--dataset", "avenue", "--old-checkpoint-dir", str(root / "old"), "--unmasked-dir", str(root / "unmasked")]
    module.main(args + ["--dry-run"])
    config = load_config(root / "config/base.yaml", [root / "config/datasets/avenue.yaml"])
    cache = VideoCache(root / "new_cache", config)
    assert cache.load(root / "good.mp4") is None  # dry run writes nothing
    module.main(args)
    loaded = cache.load(root / "good.mp4")
    assert [e.description for e in loaded.events] == ["caption 0", "caption 1", "caption 2"]
    np.testing.assert_array_equal(np.stack([e.average_embedding for e in loaded.events]), vectors)
    assert cache.load(root / "shifted.mp4") is None and cache.load(root / "missing.mp4") is None
    module.main(args)  # idempotent
    assert len(list((root / "new_cache").glob("*.npz"))) == 1


def test_incompatible_caption_settings_are_not_migrated(migrate, capsys):
    module, root, base = migrate
    video = root / "a.mp4"
    video.write_bytes(b"a")
    other = {**base, "captioning": {"mode": "middle_frame"}}
    old_checkpoint(root / "old", video, other, np.array([0]), np.array([4]))
    unmasked(root / "unmasked/avenue", "a", np.array([0]), np.array([4]), np.array([[1., 0.]], dtype=np.float32))
    module.main(["--dataset", "avenue", "--old-checkpoint-dir", str(root / "old"),
                 "--unmasked-dir", str(root / "unmasked")])
    report = json.loads(capsys.readouterr().out.split("\nVideos without")[0])
    assert report["incompatible"] == 1 and report["written"] == 0
