"""Window scores -> one score per original video frame.

Each frame takes a pooled value of the windows covering it ('mean' by default;
see METHODS), then an optional centered moving average. Centered smoothing uses
future frames, so it is for offline (whole-video) scoring.

Smoothing width in seconds = smoothing_seconds, or, when smoothing_divisor is
set, min(smoothing_seconds, video_length_seconds / smoothing_divisor).
"""

from collections import defaultdict
import math

import numpy as np
from scipy.ndimage import uniform_filter1d


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


def expand_windows(events, values, frame_count):
    """Average overlapping window values on the original frame grid."""
    values = np.asarray(values, dtype=float)
    if (not events or values.shape != (len(events),) or frame_count <= 0
            or not np.isfinite(values).all()):
        raise ValueError("Expected aligned windows and finite scalar scores")
    sums = np.zeros(frame_count + 1, dtype=float)
    counts = np.zeros(frame_count + 1, dtype=np.int64)
    for event, value in zip(events, values):
        start, stop = event.start_frame, event.stop_frame
        if stop is None or not 0 <= start < stop <= frame_count:
            raise ValueError("Invalid exclusive window boundary")
        sums[start] += value
        sums[stop] -= value
        counts[start] += 1
        counts[stop] -= 1
    coverage = np.cumsum(counts[:-1])
    if np.any(coverage == 0):
        raise ValueError("Windows leave uncovered video frames")
    return np.cumsum(sums[:-1]) / coverage


METHODS = ('mean', 'max', 'top2', 'top3', 'median', 'q75', 'q90', 'mean_max_half')


def aggregate_windows(events, values, frame_count):
    """Return frame scores for all methods with exclusive window stop bounds.

    Compute each statistic only where active windows change. Quantiles use
    NumPy's default linear interpolation. Top-k averages all available scores
    when fewer than k windows cover a frame.
    """
    values = np.asarray(values, dtype=float)
    if (not events or values.shape != (len(events),) or frame_count <= 0
            or not np.isfinite(values).all()):
        raise ValueError('Expected nonempty aligned windows and finite scores')
    changes = defaultdict(list)
    for index, event in enumerate(events):
        start, stop = event.start_frame, event.stop_frame
        if stop is None or not 0 <= start < stop <= frame_count:
            raise ValueError('Invalid exclusive window bounds')
        changes[start].append((index, True))
        changes[stop].append((index, False))
    bounds = sorted({0, frame_count, *changes})
    result = {method: np.empty(frame_count, dtype=float) for method in METHODS}
    active = set()
    for start, stop in zip(bounds, bounds[1:]):
        for index, entering in changes[start]:
            if entering:
                active.add(index)
            else:
                active.remove(index)
        if not active:
            raise ValueError('Windows leave uncovered video frames')
        ordered = np.sort(values[list(active)])
        mean, maximum = float(ordered.mean()), float(ordered[-1])
        median, q75, q90 = np.quantile(ordered, [.5, .75, .9])
        pooled = (mean, maximum, ordered[-2:].mean(), ordered[-3:].mean(),
                  median, q75, q90, .5 * (mean + maximum))
        for method, value in zip(METHODS, pooled):
            result[method][start:stop] = value
    return result


def smoothing_width(settings, frame_count, fps):
    """Moving-average width in frames (odd), or 0 for no smoothing."""
    seconds = float(settings.get("smoothing_seconds", 0.))
    divisor = settings.get("smoothing_divisor")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("smoothing_seconds must be nonnegative")
    if divisor:
        divisor = float(divisor)
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("smoothing_divisor must be positive")
        seconds = min(seconds, frame_count / fps / divisor)
    if not seconds:
        return 0
    width = max(1, int(round(seconds * fps)))
    return width + (width % 2 == 0)


def project_window_scores(events, values, frame_count, fps, settings=None, clip=True):
    """Pool covering windows, then optionally smooth the complete video.

    Window bounds are exclusive and every native frame is retained. clip=False
    keeps unbounded scores (e.g. z-scores) for a later monotonic transform.
    """
    settings = settings or {}
    method = settings.get("score_pooling", "mean")
    if method not in METHODS:
        raise ValueError(f"Unknown score pooling: {method}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")
    if method == "mean":
        scores = expand_windows(events, values, frame_count)
    else:
        scores = aggregate_windows(events, values, frame_count)[method]
    width = smoothing_width(settings, frame_count, fps)
    if width:
        scores = uniform_filter1d(scores, width, mode="nearest")
    return np.clip(scores, 0, 1) if clip else scores
