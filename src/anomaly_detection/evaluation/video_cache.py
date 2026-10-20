"""Atomic, model-versioned video features for resumable GPU preprocessing."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np

from anomaly_detection.evaluation.bank import model_spec
from anomaly_detection.utils.types import EventChunk

CACHE_VERSION = 1


class VideoCache:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.spec = model_spec(config)

    def identity(self, source):
        path = Path(source).resolve()
        stat = path.stat()
        return {"version": CACHE_VERSION, "model": self.spec, "path": str(path),
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def path(self, source):
        key = json.dumps(self.identity(source), sort_keys=True).encode()
        return self.folder / (hashlib.sha256(key).hexdigest() + ".npz")

    def load(self, source):
        path = self.path(source)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata"].item()))
                vectors = stored["embeddings"]
            if metadata["identity"] != self.identity(source):
                raise ValueError("Source/model mismatch")
            records = metadata["events"]
            if (not records or vectors.ndim != 2 or len(vectors) != len(records)
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
                    or metadata["frame_count"] <= 0 or not np.isfinite(metadata["fps"])
                    or metadata["fps"] <= 0
                    or any(not isinstance(r["description"], str) or not r["description"] for r in records)):
                raise ValueError("Invalid cached features")
            events = [EventChunk(**r, average_embedding=v) for r, v in zip(records, vectors)]
            return SimpleNamespace(events=events, frame_count=metadata["frame_count"],
                                   fps=metadata["fps"], frames=[], encoded_frames=[])
        except Exception as error:
            raise ValueError(f"Invalid checkpoint {path}; move it aside to recompute this video") from error

    def save(self, source, video, expected_identity=None):
        identity = self.identity(source)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(f"Video changed during processing: {source}")
        records = []
        for event in video.events:
            records.append({k: getattr(event, k) for k in (
                "start_frame", "end_frame", "start_time", "end_time",
                "frame_indices", "description", "stop_frame")})
        metadata = {"identity": identity, "events": records,
                    "frame_count": video.frame_count, "fps": video.fps}
        self.folder.mkdir(parents=True, exist_ok=True)
        destination = self.path(source)
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, metadata=json.dumps(metadata, allow_nan=False),
                                    embeddings=np.stack([e.average_embedding for e in video.events]))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class CachedProcessor:
    """Adapter for the unchanged serial bank builder and scoring runner."""

    def __init__(self, cache, encoder):
        self.cache, self.encoder = cache, encoder

    def process(self, source):
        result = self.cache.load(source)
        if result is None:
            raise FileNotFoundError(f"No completed video checkpoint for {source}")
        return result
