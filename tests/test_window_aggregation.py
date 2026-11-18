"""Verify offline pooling against a direct per-frame reference."""

from types import SimpleNamespace

import numpy as np
import pytest

from anomaly_detection.evaluation.window_aggregation import aggregate_windows


def test_variable_coverage_matches_direct_statistics():
    windows = [SimpleNamespace(start_frame=a, stop_frame=b)
               for a, b in [(0, 5), (1, 4), (2, 6), (3, 5)]]
    values = np.array([.1, .3, .8, .9])
    result = aggregate_windows(windows, values, 6)
    for frame in range(6):
        active = np.sort([v for w, v in zip(windows, values)
                          if w.start_frame <= frame < w.stop_frame])
        reference = dict(mean=active.mean(), max=active.max(),
                         top2=active[-2:].mean(), top3=active[-3:].mean(),
                         median=np.median(active), q75=np.quantile(active, .75),
                         q90=np.quantile(active, .9),
                         mean_max_half=.5 * (active.mean() + active.max()))
        for method, value in reference.items():
            assert result[method][frame] == pytest.approx(value)


def test_gaps_and_invalid_scores_fail():
    with pytest.raises(ValueError, match='uncovered'):
        aggregate_windows([SimpleNamespace(start_frame=1, stop_frame=3)], [.5], 3)
    with pytest.raises(ValueError):
        aggregate_windows([SimpleNamespace(start_frame=0, stop_frame=3)], [np.nan], 3)
    with pytest.raises(ValueError, match='bounds'):
        aggregate_windows([SimpleNamespace(start_frame=0, stop_frame=4)], [.5], 3)
