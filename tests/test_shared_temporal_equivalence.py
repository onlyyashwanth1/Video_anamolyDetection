"""Independent native-frame reproduction of the selected shared projection."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from anomaly_detection.inference.frame_projection import project_window_scores

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((ROOT/'config/shared_best.yaml').read_text())['temporal']


def direct_native_reference(events, values, frame_count, fps, settings):
    """Compute each frame directly, then explicitly edge-pad and convolve."""
    raw = np.empty(frame_count)
    for frame in range(frame_count):
        active = [v for e,v in zip(events,values) if e.start_frame <= frame < e.stop_frame]
        if not active:
            raise ValueError('Uncovered frame in test reference')
        raw[frame] = .5*(np.mean(active)+np.max(active))
    requested = float(settings['smoothing_seconds'])*fps
    width = max(1,int(np.rint(requested)))
    width += (width+1)%2
    radius = width//2
    return np.clip(np.convolve(np.pad(raw,(radius,radius),mode='edge'),
                              np.full(width,1/width),mode='valid'),0,1)


@pytest.mark.parametrize('fps',[.25,1.,2.5,23.976,24.,25.,29.97,30.,60.])
def test_shared_projection_matches_independent_native_reference(fps):
    # Irregular bounds, varying overlap, and a final tail with only one window.
    events = [SimpleNamespace(start_frame=a,stop_frame=b)
              for a,b in [(0,5),(1,8),(3,12),(7,15),(10,18),(14,19)]]
    values = [.01,.18,.93,.49,.82,.04]
    expected = direct_native_reference(events,values,19,fps,SETTINGS)
    actual = project_window_scores(events,values,19,fps,SETTINGS)
    assert actual.shape == (19,)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14)
    # Boundary values must use repeated edge predictions, preserving both ends.
    assert actual[0] == pytest.approx(expected[0],abs=2e-14)
    assert actual[-1] == pytest.approx(expected[-1],abs=2e-14)


@pytest.mark.parametrize('fps',[1.,23.976,29.97,60.])
@pytest.mark.parametrize('value',[0.,.123456789,.6,1.])
def test_smoothing_wider_than_clip_preserves_constant_and_tails(fps,value):
    events = [SimpleNamespace(start_frame=0,stop_frame=3),
              SimpleNamespace(start_frame=1,stop_frame=3)]
    actual = project_window_scores(events,[value,value],3,fps,SETTINGS)
    np.testing.assert_allclose(actual,np.full(3,value),rtol=0,atol=2e-14)


def test_odd_width_rounding_at_half_frame_boundary():
    events = [SimpleNamespace(start_frame=0,stop_frame=2),
              SimpleNamespace(start_frame=1,stop_frame=5)]
    # 12 seconds * 0.375 fps = 4.5 frames: nearest even is4, then odd width5.
    fps=.375
    expected=direct_native_reference(events,[.1,.9],5,fps,SETTINGS)
    actual=project_window_scores(events,[.1,.9],5,fps,SETTINGS)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14)
