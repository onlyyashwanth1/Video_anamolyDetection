"""Build normal memory from every normal video and store a reusable artifact.

The NPZ holds numeric embeddings, text, counts and JSON provenance. It requires
no pickle loading. Existing artifacts are never silently overwritten.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from anomaly_detection.evaluation.dataset import manifest
from anomaly_detection.familiarisation.domain_familiarisation import embed_constitution
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.utils.types import MemoryEntry

LOGGER = logging.getLogger(__name__)
SCHEMA = 1


def model_spec(config):
    """Versioned feature/caption contract; update when their semantics change."""
    spec = {
        "implementation": "multiscale-mean-clip_event-mean_gemma_v1",
        "encoder": config["encoder"], "video_target_fps": config["video"]["target_fps"],
        "segmentation": config["segmentation"],
        "min_caption_occurrences": config["familiarisation"]["min_caption_occurrences"],
        "caption_model": "/home/nverma/models/gemma-4-31B-it",
        "masking": "yolov8s-worldv2.pt/bytetrack/blur51",
        "strict_models": True,
    }
    from anomaly_detection.reasoning.ollama_runtime import caption_spec
    ollama = caption_spec(config)
    if ollama is not None:
        spec["caption_model"] = ollama
    if (config.get("temporal", {}).get("mode", "events") != "events"
            or config.get("captioning", {}).get("mode", "middle_frame") != "middle_frame"):
        spec.update(implementation="window_temporal_caption_v2",
                    temporal=config.get("temporal", {}), captioning=config.get("captioning", {}))
    return spec


def save_bank(path, bank, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Memory already exists: {path}. Choose a new path for a new version.")
    temporary = path.with_name(path.name + ".building")
    # Reserve our own temporary file before cleanup; never remove another build's file.
    handle = temporary.open("xb")
    try:
        with handle:
            np.savez_compressed(
                handle, embeddings=np.stack([e.embedding for e in bank.entries]),
                texts=np.asarray([e.text for e in bank.entries]),
                counts=np.asarray([e.occurrence_count for e in bank.entries]),
                metadata=np.asarray(json.dumps(metadata)),
            )
        # Hard-link publishes a complete artifact and refuses concurrent overwrite.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bank(path, config, normal_videos=None):
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        vectors, texts, counts = stored["embeddings"], stored["texts"], stored["counts"]
    if metadata.get("schema") != SCHEMA or metadata.get("model_spec") != model_spec(config):
        raise ValueError("Memory was built with a different model/configuration; build a new version")
    if normal_videos is not None and metadata["normal_videos"] != manifest(normal_videos):
        raise ValueError("Normal-video manifest changed; choose a new memory artifact path")
    if (vectors.ndim != 2 or len(vectors) == 0 or len(vectors) != len(texts)
            or len(counts) != len(texts) or not np.isfinite(vectors).all()
            or np.any(counts <= 0) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Invalid memory artifact arrays")
    entries = [MemoryEntry(str(t), v, int(c)) for t, v, c in zip(texts, vectors, counts)]
    return MemoryBank(entries), metadata


def build_bank(processor, normal_videos, config, path):
    if Path(path).exists():
        raise FileExistsError(f"Memory already exists: {path}; use it or choose a new path")
    counts, sources = Counter(), []
    for index, (name, video_path) in enumerate(normal_videos.items(), 1):
        LOGGER.info("Memory [%d/%d] %s", index, len(normal_videos), name)
        try:
            video = processor.process(str(video_path))
        except Exception as error:
            raise RuntimeError(f"Normal-memory processing failed for {video_path}") from error
        counts.update(event.description for event in video.events)
        sources.append({"name": name, "events": len(video.events),
                        "frames": video.frame_count, "fps": video.fps})
        del video
    minimum = config["familiarisation"]["min_caption_occurrences"]
    texts = [text for text, count in counts.items() if count >= minimum]
    if not texts:
        raise ValueError("No captions survived normal-memory filtering")
    bank = MemoryBank(embed_constitution(texts, processor.encoder, counts))
    metadata = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_spec": model_spec(config), "normal_videos": manifest(normal_videos),
        "processed_videos": sources, "entry_count": len(bank),
        "normal_video_count": len(normal_videos), "config": config,
    }
    save_bank(path, bank, metadata)
    LOGGER.info("Permanent normal memory saved to %s (%d entries)", path, len(bank))
    return bank, metadata


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
