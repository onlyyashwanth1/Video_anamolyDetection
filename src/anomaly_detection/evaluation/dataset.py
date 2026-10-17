"""Video manifests, split checks and explicit annotation/frame alignment."""

from dataclasses import dataclass
from pathlib import Path
import math
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
NATIVE_END_TOLERANCE = 2


@dataclass(frozen=True)
class Annotation:
    name: str
    category: str
    ranges: tuple  # one-based, inclusive annotation frame indices


def discover_videos(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Video folder does not exist: {folder}")
    videos = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            if path.stem in videos:
                raise ValueError(f"Duplicate video identifier: {path.stem}")
            videos[path.stem] = path
    if not videos:
        raise ValueError(f"No videos found in {folder}")
    return videos


def manifest(videos):
    return [{"name": name, "path": str(path.resolve()), "bytes": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns} for name, path in sorted(videos.items())]


def check_disjoint(normal, testing):
    overlap = set(normal) & set(testing)
    overlap_paths = {p.resolve() for p in normal.values()} & {p.resolve() for p in testing.values()}
    if overlap or overlap_paths:
        raise ValueError(f"Normal-memory/test overlap: {sorted(overlap)} {sorted(map(str, overlap_paths))}")


def load_annotations(path):
    records = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4 or (len(parts) - 2) % 2:
            raise ValueError(f"Invalid annotation row {number}: {line}")
        name, category = Path(parts[0]).stem, parts[1]
        if name in records:
            raise ValueError(f"Duplicate annotation: {name}")
        numbers = list(map(int, parts[2:]))
        ranges = []
        for start, end in zip(numbers[::2], numbers[1::2]):
            if (start, end) == (-1, -1):
                continue
            if not 1 <= start <= end:
                raise ValueError(f"Invalid interval for {name}: {start}, {end}")
            ranges.append((start, end))
        if (category.lower() == "normal") != (not ranges):
            raise ValueError(f"Category and intervals disagree for {name}")
        records[name] = Annotation(name, category, tuple(ranges))
    if not records:
        raise ValueError("Annotation file is empty")
    return records


def validate_test_manifest(videos, annotations):
    missing = sorted(set(annotations) - set(videos))
    extra = sorted(set(videos) - set(annotations))
    if missing or extra:
        raise ValueError(f"Test folder/annotation mismatch. Missing videos: {missing}; unannotated: {extra}")


def frame_labels(annotation, frame_count, video_fps, annotation_fps="native"):
    """Map annotation frames onto ALL original video frames, never 1 FPS samples.

    'native' means annotation indices already reference the original video.
    Numeric annotation FPS is only for intervals indexed on a resampled copy.
    Ranges are one-based inclusive; output arrays are zero-based.
    Native endpoints up to two frames past EOF are clipped with a warning,
    provided the interval starts inside the video. Larger overruns fail.
    """
    rate = video_fps if annotation_fps == "native" else float(annotation_fps)
    if frame_count <= 0 or not math.isfinite(rate) or rate <= 0 or video_fps <= 0:
        raise ValueError("Invalid frame count or annotation/video FPS")
    limit = frame_count if annotation_fps == "native" else math.ceil(frame_count * rate / video_fps)
    labels = np.zeros(frame_count, dtype=np.uint8)
    for start, end in annotation.ranges:
        if end > limit:
            if (annotation_fps == "native" and start <= limit
                    and end - limit <= NATIVE_END_TOLERANCE):
                LOGGER.warning(
                    "%s: clipping annotation end %d to last frame %d (%d-frame EOF overrun)",
                    annotation.name, end, limit, end - limit,
                )
                end = limit
            else:
                raise ValueError(
                    f"{annotation.name}: annotation end {end} exceeds {limit} frames at "
                    f"annotation_fps={annotation_fps}. Check the annotation time base."
                )
        first = math.ceil((start - 1) * video_fps / rate - 1e-9)
        stop = min(frame_count, math.ceil(end * video_fps / rate - 1e-9))
        labels[first:stop] = 1
    return labels


def expand_events(events, values, frame_count):
    """Hold each event's value until the next event starts, covering the tail.

    Event end_frame is the last sampled frame, so using it as an exclusive
    bound would leave gaps between samples and omit the final video frames.
    """
    starts = [event.start_frame for event in events]
    if (not starts or starts[0] != 0 or len(events) != len(values)
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or starts[-1] >= frame_count):
        raise ValueError("Events must cover the video from frame zero in increasing order")
    result = np.empty(frame_count, dtype=np.asarray(values).dtype)
    for start, stop, value in zip(starts, starts[1:] + [frame_count], values):
        result[start:stop] = value
    return result
