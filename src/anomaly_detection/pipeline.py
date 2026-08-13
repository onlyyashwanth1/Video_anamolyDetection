"""
Pipeline orchestrator - wires together every module in the order
described in the paper. Currently, Steps 1-3 are fully functional;
later steps are stubbed (see docs/PROJECT_STATUS.md) and will raise
NotImplementedError if called - the commented-out code at the bottom
shows exactly how they'll plug in once built.
"""

import math
from collections import Counter
from typing import List, Tuple
import yaml

from anomaly_detection.ingestion.video_stream import VideoStream
from anomaly_detection.encoding.clip_encoder import ClipEncoder
from anomaly_detection.segmentation.temporal_decomposition import build_event_tree
from anomaly_detection.familiarisation.domain_familiarisation import (
    caption_event, build_domain_constitution, embed_constitution,
)
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.utils.types import EncodedFrame, EventChunk, AnomalyResult


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_steps_1_to_3(video_source, config: dict):
    """
    Runs Steps 1-3: video in -> CLIP fingerprints -> event tree.
    Each returned EventChunk has .average_embedding set, ready for
    familiarisation/inference to use.
    """
    stream = VideoStream(source=video_source, target_fps=config["video"]["target_fps"])
    encoder = ClipEncoder(model_name=config["encoder"]["model_name"])

    encoded_frames: List[EncodedFrame] = [
        encoder.encode_frame(frame) for frame in stream.frames()
    ]

    tree = build_event_tree(
        encoded_frames,
        coarse_threshold=config["segmentation"]["coarse_threshold"],
        fine_threshold=config["segmentation"]["fine_threshold"],
    )
    return tree, encoder, [ef.frame for ef in encoded_frames]


def run_full_pipeline(video_source, config: dict) -> Tuple[List[str], MemoryBank, List[AnomalyResult]]:
    """
    Runs Steps 1-10 end to end on a single video/webcam source.
    """
    tree, encoder, raw_frames = run_steps_1_to_3(video_source, config)

    if len(tree) < 2:
        raise ValueError(
            f"Only found {len(tree)} top-level event(s) - need at least 2 "
            "(some for familiarisation, at least one to test as 'live'). "
            "Try a longer video or a lower segmentation.coarse_threshold."
        )

    # Split events: first N% for familiarisation, the rest treated as "live"
    split_index = max(1, math.ceil(len(tree) * config["familiarisation"]["observation_fraction"]))
    split_index = min(split_index, len(tree) - 1)  # always leave at least 1 live event
    familiarisation_events = tree[:split_index]
    live_events = tree[split_index:]

    # --- Steps 5-6: build the "Normal" notebook ---
    captions = [caption_event(e, encoder, raw_frames=raw_frames) for e in familiarisation_events]
    for event, caption in zip(familiarisation_events, captions):
        event.description = caption

    counts = Counter(captions)
    constitution = build_domain_constitution(
        captions, min_occurrences=config["familiarisation"]["min_caption_occurrences"]
    )
    memory_entries = embed_constitution(constitution, encoder, occurrence_counts=counts)

    # --- Step 7: initialize the memory bank ---
    memory_bank = MemoryBank(initial_entries=memory_entries)

    # --- Steps 8-10: score every live event against the notebook & rationalize anomalies ---
    threshold = config["inference"]["anomaly_threshold"]
    results: List[AnomalyResult] = []
    for event in live_events:
        event.description = caption_event(event, encoder, raw_frames=raw_frames)  # caption every live event too
        result = score_event(event, memory_bank, threshold)

        if result.is_anomaly:
            result = explain_anomaly(result, constitution)
            memory_bank.log_unmatched(event.description, event.average_embedding)
            memory_bank.try_promote(event.description, config["memory"]["promotion_repeat_count"])

        results.append(result)

    return constitution, memory_bank, results
