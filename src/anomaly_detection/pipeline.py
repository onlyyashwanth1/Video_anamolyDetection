"""
Pipeline orchestrator - wires together every module in the order
described in the paper. Currently, Steps 1-3 are fully functional;
later steps are stubbed (see docs/PROJECT_STATUS.md) and will raise
NotImplementedError if called - the commented-out code at the bottom
shows exactly how they'll plug in once built.
"""

from typing import List
import yaml

from anomaly_detection.ingestion.video_stream import VideoStream
from anomaly_detection.encoding.clip_encoder import ClipEncoder
from anomaly_detection.segmentation.temporal_decomposition import build_event_tree
from anomaly_detection.utils.types import EncodedFrame, EventChunk


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_steps_1_to_3(video_source, config: dict) -> List[EventChunk]:
    """
    Runs the currently-implemented part of the pipeline:
        video in -> CLIP fingerprints -> event tree
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
    return tree


# ---------------------------------------------------------------------------
# Steps 4-10 will be wired in here as each module is implemented.
# This is intentionally left as a sketch so it's obvious where new calls go:
#
#   from anomaly_detection.masking.object_masking import mask_frame
#   from anomaly_detection.familiarisation.domain_familiarisation import (
#       caption_event, build_domain_constitution, embed_constitution)
#   from anomaly_detection.memory.memory_bank import MemoryBank
#   from anomaly_detection.inference.adaptive_inference import score_event
#   from anomaly_detection.reasoning.llm_reasoning import explain_anomaly
#
#   def run_full_pipeline(video_source, config: dict):
#       # Step 4: mask each frame before encoding
#       ...
#       # Steps 5-6: during the observation window, build the notebook
#       ...
#       # Step 7: initialize the memory bank from the notebook
#       memory_bank = MemoryBank(initial_entries=...)
#       # Steps 8-9: score every subsequent event
#       results = [score_event(e, ..., memory_bank, config["inference"]["anomaly_threshold"])
#                  for e in tree]
#       # Step 10: explain only the flagged ones
#       results = [explain_anomaly(r, notebook) if r.is_anomaly else r for r in results]
#       return results
# ---------------------------------------------------------------------------
