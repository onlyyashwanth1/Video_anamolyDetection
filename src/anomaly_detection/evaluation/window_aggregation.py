"""Offline alternatives to mean pooling of overlapping, fixed-score windows."""

from collections import defaultdict

import numpy as np


METHODS = ('mean', 'max', 'top2', 'top3', 'median', 'q75', 'q90', 'mean_max_half')


def aggregate_windows(events, values, frame_count):
    """Return frame scores for all methods with exclusive window stop bounds.

    Compute each statistic only where active windows change. Quantiles use
    NumPy's default linear interpolation. Top-k averages all available scores
    when fewer than k windows cover a frame.
    """
    values = np.asarray(values, dtype=float)
    if (not events or values.shape != (len(events),) or frame_count <= 0
            or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        raise ValueError('Expected nonempty aligned windows and scores in [0, 1]')
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
