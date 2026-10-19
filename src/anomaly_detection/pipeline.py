"""
Pipeline orchestrator - wires together every module in the order described in the paper.
"""

import math
from collections import Counter
from typing import List, Tuple
import yaml

from anomaly_detection.processing import VideoEventProcessor
from anomaly_detection.familiarisation.domain_familiarisation import (
    build_domain_constitution, embed_constitution,
)
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.reasoning.llm_reasoning import explain_anomaly
from anomaly_detection.utils.types import AnomalyResult


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_steps_1_to_3(video_source, config: dict):
    processor = VideoEventProcessor(config)
    video = processor.process(video_source, caption=False)
    return video.events, processor.encoder, video.frames, video.encoded_frames


def run_full_pipeline(video_source, config: dict) -> Tuple[List[str], MemoryBank, List[AnomalyResult]]:
    processor = VideoEventProcessor(config)
    video = processor.process(video_source, caption=False)
    tree, encoder, raw_frames = video.events, processor.encoder, video.frames

    if len(tree) < 2:
        raise ValueError(
            f"Only found {len(tree)} top-level event(s) - need at least 2."
        )

    if config.get("temporal", {}).get("mode", "events") == "sliding_windows":
        from anomaly_detection.segmentation.sliding_windows import split_familiarisation
        familiarisation_events, live_events = split_familiarisation(
            tree, config["familiarisation"]["observation_fraction"], video.frame_count)
    else:
        split_index = max(1, math.ceil(len(tree) * config["familiarisation"]["observation_fraction"]))
        split_index = min(split_index, len(tree) - 1)
        familiarisation_events, live_events = tree[:split_index], tree[split_index:]

    processor.caption_events(familiarisation_events, raw_frames)
    processor.caption_events(live_events, raw_frames)
    captions = [event.description for event in familiarisation_events]
    counts = Counter(captions)
    constitution = build_domain_constitution(
        captions, min_occurrences=config["familiarisation"]["min_caption_occurrences"]
    )
    memory_entries = embed_constitution(constitution, encoder, occurrence_counts=counts)

    memory_bank = MemoryBank(initial_entries=memory_entries)

    threshold = config["inference"]["anomaly_threshold"]
    action_scorer = None
    if config["inference"].get("scoring_mode", "memory") != "memory":
        from anomaly_detection.inference.action_scoring import ActionScorer
        action_scorer = ActionScorer(encoder, config["inference"].get("action", {}))
    results: List[AnomalyResult] = []
    for event in live_events:
        result = score_event(event, memory_bank, threshold, encoder=encoder,
                             action_scorer=action_scorer, settings=config["inference"])

        if result.is_anomaly:
            result = explain_anomaly(result, constitution)
            if result.is_anomaly and config["memory"].get("allow_anomaly_promotion", False):
                memory_bank.log_unmatched(event.description, encoder.encode_text(event.description))
                memory_bank.try_promote(event.description, config["memory"]["promotion_repeat_count"])

        results.append(result)

    return constitution, memory_bank, results
