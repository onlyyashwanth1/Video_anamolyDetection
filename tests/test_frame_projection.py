"""Projection preserves boundaries and uses the same durations at different FPS."""

from types import SimpleNamespace

import numpy as np
import pytest

from anomaly_detection.inference.frame_projection import project_window_scores


def test_smoothing_matches_explicit_padded_convolution():
    events = [SimpleNamespace(start_frame=0, stop_frame=4),
              SimpleNamespace(start_frame=2, stop_frame=6)]
    raw = np.array([.2, .2, .65, .65, .8, .8])
    # Mean-max blend with two covering windows: .5*(.5+.8)=.65.
    settings = dict(score_pooling='mean_max_half', smoothing_seconds=2.)
    result = project_window_scores(events, [.2, .8], 6, 2., settings)
    expected = np.convolve(np.pad(raw, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('fps', [1., 23.976, 24., 30.])
def test_long_smoothing_preserves_constant_short_video(fps):
    events = [SimpleNamespace(start_frame=0, stop_frame=3)]
    result = project_window_scores(events, [.6], 3, fps,
                                   dict(score_pooling='mean_max_half', smoothing_seconds=12.))
    np.testing.assert_allclose(result, [.6]*3)


def test_invalid_smoothing_fails():
    with pytest.raises(ValueError):
        project_window_scores([], [], 1, 24, {'smoothing_seconds': float('nan')})
