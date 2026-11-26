"""Build, save and load the saved normal-memory bank (NPZ, no pickle).

The bank is built from a dataset's normal training videos: 1 - calibration_fraction
of them become reference exemplars (window CLIP embeddings + caption entries);
the rest are held out to fit the distance calibration. Artifacts are never
overwritten, and loading refuses a bank built with different features or settings.

feature_spec() is the perception contract shared with the video cache
(evaluation/video_cache.py); change its version when feature semantics change.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from anomaly_detection.datatypes import MemoryEntry
from anomaly_detection.evaluation.dataset import manifest

LOGGER = logging.getLogger(__name__)
SCHEMA = 3
FEATURE_VERSION = "window_caption_unmasked_v3"


def feature_spec(config):
    """What determines cached window embeddings and captions."""
    spec = {
        "implementation": FEATURE_VERSION,
        "image_preprocessing": "none",
        "encoder": config["encoder"], "video_target_fps": config["video"]["target_fps"],
        "temporal": {k: config.get("temporal", {}).get(k) for k in ("mode", "window_seconds", "stride_seconds")},
        "captioning": config.get("captioning", {}),
        "caption_model": "transformers:gemma-4-31B-it",
        "strict_models": True,
    }
    if spec["temporal"]["mode"] in (None, "events"):
        spec["segmentation"] = config["segmentation"]
    from anomaly_detection.reasoning.ollama_runtime import caption_spec
    ollama = caption_spec(config)
    if ollama is not None:
        spec["caption_model"] = ollama
    return spec


def memory_spec(config):
    settings = config.get("memory_bank", {})
    return {"version": 3,
            "min_caption_occurrences": int(settings.get("min_caption_occurrences", 1)),
            "calibration_fraction": float(settings.get("calibration_fraction", .2)),
            "calibration_seed": int(settings.get("calibration_seed", 42)),
            "calibration_windows_per_video": int(settings.get("calibration_windows_per_video", 256)),
            "top_k": int(settings.get("top_k", 10))}


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
                visual_embeddings=bank.visual_embeddings,
                visual_text_indices=bank.visual_text_indices,
                visual_sources=bank.visual_sources, visual_starts=bank.visual_starts,
            )
        # Hard-link publishes a complete artifact and refuses concurrent overwrite.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bank(path, config, normal_videos=None):
    from anomaly_detection.memory.visual_text import VisualTextMemoryBank

    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"{path} uses bank schema {metadata.get('schema')}; this code reads schema "
                             f"{SCHEMA} (unmasked features). Build a new bank.")
        vectors, texts, counts = stored["embeddings"], stored["texts"], stored["counts"]
        visual = {key: stored[key] for key in ("visual_embeddings", "visual_text_indices",
                                                "visual_sources", "visual_starts")}
    if metadata.get("feature_spec") != feature_spec(config):
        raise ValueError("Memory was built with different features/captions; build a new version")
    if metadata.get("memory_spec") != memory_spec(config):
        raise ValueError("Memory bank settings differ from the configuration; build a new bank")
    if normal_videos is not None and metadata["normal_videos"] != manifest(normal_videos):
        raise ValueError("Normal-video manifest changed; choose a new memory artifact path")
    if (vectors.ndim != 2 or len(vectors) == 0 or len(vectors) != len(texts)
            or len(counts) != len(texts) or not np.isfinite(vectors).all()
            or np.any(counts <= 0) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Invalid memory artifact arrays")
    entries = [MemoryEntry(str(t), v, int(c)) for t, v, c in zip(texts, vectors, counts)]
    bank = VisualTextMemoryBank(entries, calibration=metadata["calibration"],
                                device=config.get("memory_bank", {}).get("search_device", "cpu"),
                                top_k=metadata["memory_spec"]["top_k"], **visual)
    return bank, metadata


def normal_partition(normal_videos, fraction, seed):
    if len(normal_videos) < 2 or not 0 < fraction < 1:
        raise ValueError("The bank needs >=2 normal videos and calibration_fraction in (0,1)")
    names = np.asarray(sorted(normal_videos))
    shuffled = np.random.default_rng(seed).permutation(names)
    count = min(len(names) - 1, max(1, round(len(names) * fraction)))
    calibration = set(shuffled[:count])
    return ({n: normal_videos[n] for n in sorted(normal_videos) if n not in calibration},
            {n: normal_videos[n] for n in sorted(calibration)})


def build_bank(processor, normal_videos, config, path, text_lookup=None):
    """Build paired image/caption exemplars and fit their calibration.

    text_lookup optionally maps caption -> CLIP text embedding to skip re-encoding.
    """
    from anomaly_detection.memory.visual_text import NeighborIndex, VisualTextMemoryBank, fit_calibration

    if Path(path).exists():
        raise FileExistsError(f"Memory already exists: {path}; choose a new path")
    spec = memory_spec(config)
    if spec["top_k"] < 1 or spec["calibration_windows_per_video"] < 1:
        raise ValueError("top_k and calibration_windows_per_video must be positive")
    reference, calibration = normal_partition(normal_videos, spec["calibration_fraction"],
                                             spec["calibration_seed"])
    text_lookup = text_lookup or {}
    texts, text_indices, counts, vectors, indices, sources, starts = [], {}, Counter(), [], [], [], []
    for number, (name, source) in enumerate(reference.items(), 1):
        LOGGER.info("Memory bank [%d/%d] %s", number, len(reference), name)
        video = processor.process(str(source))
        for event in video.events:
            text = event.description
            if text not in text_indices:
                text_indices[text] = len(texts)
                texts.append(text)
            counts[text] += 1
            vectors.append(event.average_embedding)
            indices.append(text_indices[text])
            sources.append(name)
            starts.append(event.start_frame)
    keep = [i for i, text in enumerate(texts) if counts[text] >= spec["min_caption_occurrences"]]
    if not keep:
        raise ValueError("No normal memory captions survived filtering")
    remap = {old: new for new, old in enumerate(keep)}
    visual_keep = [i for i, text_index in enumerate(indices) if text_index in remap]
    entries = [MemoryEntry(texts[i], text_lookup[texts[i]] if texts[i] in text_lookup
                           else processor.encoder.encode_text(texts[i]), counts[texts[i]]) for i in keep]
    visual_vectors = np.asarray(vectors, dtype=np.float32)[visual_keep]
    text_vectors = np.stack([e.embedding for e in entries])
    device = config.get("memory_bank", {}).get("search_device", "cpu")
    image_index, text_index = NeighborIndex(visual_vectors, device), NeighborIndex(text_vectors, device)
    normal_images, normal_texts = [], []
    for name, source in calibration.items():
        video = processor.process(str(source))
        # Equal per-video cap prevents the longest clips overwhelming calibration.
        chosen = np.linspace(0, len(video.events) - 1,
                             min(len(video.events), spec["calibration_windows_per_video"]), dtype=int)
        for i in chosen:
            event = video.events[i]
            normal_images.append(event.average_embedding)
            normal_texts.append(text_lookup[event.description] if event.description in text_lookup
                                else processor.encoder.encode_text(event.description))
    LOGGER.info("Calibrating memory distances on %d held-out normal windows", len(normal_images))
    parameters = {"visual": fit_calibration(image_index.distances(np.asarray(normal_images), spec["top_k"])),
                  "text": fit_calibration(text_index.distances(np.asarray(normal_texts), spec["top_k"]))}
    bank = VisualTextMemoryBank(entries, visual_vectors, [remap[indices[i]] for i in visual_keep],
                                [sources[i] for i in visual_keep], [starts[i] for i in visual_keep],
                                parameters, device, spec["top_k"])
    bank._indices = (image_index, text_index)  # reuse the indices built for calibration
    metadata = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "feature_spec": feature_spec(config), "memory_spec": spec,
                "normal_videos": manifest(normal_videos), "reference_videos": manifest(reference),
                "calibration_videos": manifest(calibration), "normal_video_count": len(reference),
                "entry_count": len(entries), "visual_entry_count": len(visual_vectors),
                "calibration": parameters, "config": config}
    save_bank(path, bank, metadata)
    LOGGER.info("Normal memory saved to %s (%d windows, %d captions)", path, len(visual_vectors), len(entries))
    return bank, metadata


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
