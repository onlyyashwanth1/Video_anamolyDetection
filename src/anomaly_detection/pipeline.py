"""
Pipeline orchestrator - wires together every module in the order described in the paper.
"""

import math
from collections import Counter
from typing import List, Tuple
import yaml

from anomaly_detection.processing import VideoEventProcessor
from anomaly_detection.familiarisation.domain_familiarisation import (
    caption_event, build_domain_constitution, embed_constitution,
)
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.reasoning.llm_reasoning import explain_anomaly
from anomaly_detection.utils.types import EncodedFrame, EventChunk, AnomalyResult


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_steps_1_to_3(video_source, config: dict):
    processor = VideoEventProcessor(config)
    video = processor.process(video_source, caption=False)
    return video.events, processor.encoder, video.frames, video.encoded_frames


def run_full_pipeline(video_source, config: dict) -> Tuple[List[str], MemoryBank, List[AnomalyResult]]:
    tree, encoder, raw_frames, encoded_frames = run_steps_1_to_3(video_source, config)

    if len(tree) < 2:
        raise ValueError(
            f"Only found {len(tree)} top-level event(s) - need at least 2."
        )

    split_index = max(1, math.ceil(len(tree) * config["familiarisation"]["observation_fraction"]))
    split_index = min(split_index, len(tree) - 1)
    familiarisation_events = tree[:split_index]
    live_events = tree[split_index:]

    # Caption familiarisation events using local Gemma VLM
    captions = [caption_event(e, encoder, raw_frames=raw_frames) for e in familiarisation_events]
    for event, caption in zip(familiarisation_events, captions):
        event.description = caption

    counts = Counter(captions)
    constitution = build_domain_constitution(
        captions, min_occurrences=config["familiarisation"]["min_caption_occurrences"]
    )
    memory_entries = embed_constitution(constitution, encoder, occurrence_counts=counts)

    memory_bank = MemoryBank(initial_entries=memory_entries)

    threshold = config["inference"]["anomaly_threshold"]
    results: List[AnomalyResult] = []
    for event in live_events:
        event.description = caption_event(event, encoder, raw_frames=raw_frames)
        result = score_event(event, memory_bank, threshold, encoder=encoder)

        if result.is_anomaly:
            result = explain_anomaly(result, constitution)
            if result.is_anomaly:
                memory_bank.log_unmatched(event.description, event.average_embedding)
                memory_bank.try_promote(event.description, config["memory"]["promotion_repeat_count"])

        results.append(result)

    return constitution, memory_bank, results
