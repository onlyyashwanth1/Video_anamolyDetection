"""Data types shared by every stage of the pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Frame:
    """One sampled video frame (perception/video.py)."""
    frame_number: int
    timestamp_sec: float
    image: np.ndarray  # RGB, shape (H, W, 3)


@dataclass
class EncodedFrame:
    """A Frame plus its CLIP image embedding (perception/clip_encoder.py)."""
    frame: Frame
    embedding: np.ndarray  # normalized 1D vector


@dataclass
class EventChunk:
    """A scored unit of video: a sliding window (perception/windows.py) or an
    event from the similarity-drop tree (perception/events.py)."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frame_indices: List[int] = field(default_factory=list)
    children: List["EventChunk"] = field(default_factory=list)
    description: Optional[str] = None  # Gemma caption (perception/captioner.py)
    average_embedding: Optional[np.ndarray] = None  # mean CLIP embedding of member frames
    stop_frame: Optional[int] = None  # exclusive native-frame boundary for windows

    def __repr__(self):
        return (f"EventChunk(frames {self.start_frame}-{self.end_frame}, "
                f"time {self.start_time:.1f}s-{self.end_time:.1f}s, "
                f"{len(self.children)} sub-events)")


@dataclass
class MemoryEntry:
    """One normal caption in memory, with its text embedding and frequency."""
    text: str
    embedding: np.ndarray
    occurrence_count: int = 1


@dataclass
class AnomalyResult:
    """Score and decision for one window or event (scoring/), optionally with a
    Gemma explanation (reasoning/explain.py)."""
    event: EventChunk
    anomaly_score: float
    is_anomaly: bool
    closest_match: Optional[str]
    explanation: Optional[str] = None
    component_scores: dict = field(default_factory=dict)
