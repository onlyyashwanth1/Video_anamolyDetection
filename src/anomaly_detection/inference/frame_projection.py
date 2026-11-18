"""Shared native-frame projection for offline action-window predictions."""

import math

import numpy as np
from scipy.ndimage import uniform_filter1d

from anomaly_detection.evaluation.dataset import expand_windows
from anomaly_detection.evaluation.window_aggregation import METHODS, aggregate_windows


def project_window_scores(events, values, frame_count, fps, settings=None):
    """Pool covering windows, then optionally smooth the complete video.

    Centered smoothing uses future frames and is intended for offline video
    evaluation. Window bounds are exclusive and every native frame is retained.
    """
    settings = settings or {}
    method = settings.get('score_pooling', 'mean')
    seconds = float(settings.get('smoothing_seconds', 0.))
    if method not in METHODS:
        raise ValueError(f'Unknown score pooling: {method}')
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('FPS must be positive and smoothing seconds nonnegative')
    if method == 'mean':
        scores = expand_windows(events, values, frame_count)
    else:
        scores = aggregate_windows(events, values, frame_count)[method]
    if seconds:
        width = max(1, int(round(seconds * fps)))
        if width % 2 == 0:
            width += 1
        scores = uniform_filter1d(scores, width, mode='nearest')
    return np.clip(scores, 0, 1)
