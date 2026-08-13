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


def run_steps_1_to_3(video_source, config: dict) -> List[EventChunk]:
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
    return tree, encoder


def run_full_pipeline(video_source, config: dict) -> Tuple[List[str], MemoryBank, List[AnomalyResult]]:
    """
    Runs Steps 1-9 end to end on a single video/webcam source:

        video in -> fingerprints -> event tree                     (Steps 1-3)
        -> caption the first `observation_fraction` of events      (Step 5)
        -> build + embed the "Normal" notebook from them            (Steps 5-6)
        -> initialize the memory bank                                (Step 7)
        -> score every remaining event against the notebook          (Steps 8-9)

    Step 4 (masking) and Step 10 (LLM explanation) are not yet wired in -
    see docs/PROJECT_STATUS.md.

    Returns: (notebook_sentences, memory_bank, results_for_live_events)
    """
    tree, encoder = run_steps_1_to_3(video_source, config)

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
    captions = [caption_event(e, encoder) for e in familiarisation_events]
    for event, caption in zip(familiarisation_events, captions):
        event.description = caption

    counts = Counter(captions)
    constitution = build_domain_constitution(
        captions, min_occurrences=config["familiarisation"]["min_caption_occurrences"]
    )
    memory_entries = embed_constitution(constitution, encoder, occurrence_counts=counts)

    # --- Step 7: initialize the memory bank ---
    memory_bank = MemoryBank(initial_entries=memory_entries)

    # --- Steps 8-9: score every live event against the notebook ---
    threshold = config["inference"]["anomaly_threshold"]
    results: List[AnomalyResult] = []
    for event in live_events:
        event.description = caption_event(event, encoder)  # caption every live event too, for readable output
        result = score_event(event, memory_bank, threshold)
        results.append(result)

        # Demonstrates the Step 7 familiarity-counter mechanism: recurring
        # anomalies eventually get promoted into "normal" instead of being
        # flagged forever. On a single short demo run this rarely triggers
        # (not enough repeats), but the logic is real and testable.
        if result.is_anomaly:
            memory_bank.log_unmatched(event.description, event.average_embedding)
            memory_bank.try_promote(event.description, config["memory"]["promotion_repeat_count"])

    return constitution, memory_bank, results
