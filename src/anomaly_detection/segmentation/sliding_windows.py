"""Time-based overlapping windows with explicit native-frame coverage."""

import math
import numpy as np

from anomaly_detection.utils.types import EventChunk


def build_sliding_windows(encoded_frames, frame_count, fps,
                          window_seconds=4.0, stride_seconds=1.0):
    if (not encoded_frames or frame_count <= 0 or not math.isfinite(fps) or fps <= 0
            or not 0 < stride_seconds <= window_seconds
            or not math.isfinite(window_seconds)):
        raise ValueError("Need frames and finite positive window/stride with stride <= window")
    width = max(1, round(window_seconds * fps))
    stride = max(1, round(stride_seconds * fps))
    positions = np.asarray([f.frame.frame_number for f in encoded_frames])
    if positions[0] != 0 or np.any(np.diff(positions) <= 0) or positions[-1] >= frame_count:
        raise ValueError("Sampled frames must start at zero and increase within the video")
    windows = []
    for start in range(0, frame_count, stride):
        stop = min(start + width, frame_count)
        indices = np.flatnonzero((positions >= start) & (positions < stop)).tolist()
        if not indices:
            raise ValueError("Window contains no sampled frames; increase sampling FPS or window size")
        mean = np.mean([encoded_frames[i].embedding for i in indices], axis=0)
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid window embedding")
        windows.append(EventChunk(
            start_frame=start, end_frame=int(positions[indices[-1]]),
            start_time=start / fps, end_time=stop / fps, stop_frame=stop,
            frame_indices=indices, average_embedding=mean / norm,
        ))
        if stop == frame_count:
            break
    return windows


def split_familiarisation(events, fraction, frame_count):
    """Exclude windows crossing the demo's observation boundary."""
    if not 0 < fraction < 1:
        raise ValueError("observation_fraction must be between zero and one")
    boundary = int(frame_count * fraction)
    normal = [e for e in events if e.stop_frame <= boundary]
    live = [e for e in events if e.start_frame >= boundary]
    if not normal or not live:
        raise ValueError("Video too short for disjoint observation and inference windows")
    return normal, live
