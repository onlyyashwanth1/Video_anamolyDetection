"""
Shared data types used across every module in the pipeline.

Keeping these in one place means every module agrees on the exact same
shape of data, instead of each file inventing its own slightly-different
frame/event structure. If you add a new field later (e.g. a masked image),
add it here once and every module that imports Frame/EventChunk sees it.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Frame:
    """One sampled video frame. Produced by: ingestion/video_stream.py (Step 1)."""
    frame_number: int
    timestamp_sec: float
    image: np.ndarray  # RGB, shape (H, W, 3)


@dataclass
class EncodedFrame:
    """A Frame plus its CLIP fingerprint. Produced by: encoding/clip_encoder.py (Step 2)."""
    frame: Frame
    embedding: np.ndarray  # normalized 1D vector


@dataclass
class EventChunk:
    """
    A group of consecutive frames treated as one 'event', possibly with
    finer sub-events nested inside. Produced by:
    segmentation/temporal_decomposition.py (Step 3/4 - GEBD + HGTree).
    """
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frame_indices: List[int] = field(default_factory=list)
    children: List["EventChunk"] = field(default_factory=list)
    description: Optional[str] = None  # filled in later by familiarisation/reasoning steps

    def __repr__(self):
        return (f"EventChunk(frames {self.start_frame}-{self.end_frame}, "
                f"time {self.start_time:.1f}s-{self.end_time:.1f}s, "
                f"{len(self.children)} sub-events)")


@dataclass
class MemoryEntry:
    """
    One line in the 'Normal' notebook (Domain Constitution). Produced by:
    familiarisation/domain_familiarisation.py (Steps 5-6),
    updated by: memory/memory_bank.py (Step 7 + familiarity counter).
    """
    text: str
    embedding: np.ndarray
    occurrence_count: int = 1


@dataclass
class AnomalyResult:
    """
    Final verdict for one event. Produced by:
    inference/adaptive_inference.py (Steps 8-9 - similarity + threshold),
    filled in further by: reasoning/llm_reasoning.py (Step 10 - explanation).
    """
    event: EventChunk
    anomaly_score: float
    is_anomaly: bool
    closest_match: Optional[str]
    explanation: Optional[str] = None
