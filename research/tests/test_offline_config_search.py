"""Independent references for the compressed offline frame-metric search."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_detection.evaluation.window_aggregation import aggregate_windows

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/search_xd_configs.py'
SPEC = importlib.util.spec_from_file_location('offline_config_search', SCRIPT)
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


def test_weighted_segment_metrics_equal_expanded_frames_with_ties():
    # Include all-positive, all-negative and mixed segments, with multiple
    # segments at the exact same score (the AP tie handling is consequential).
    scores = np.array([.2, .2, .5, .5, .8, .9, .9])
    positive = np.array([0, 3, 4, 0, 2, 5, 1])
    negative = np.array([5, 2, 0, 7, 4, 0, 3])
    y = np.concatenate([np.r_[np.ones(p), np.zeros(n)] for p,n in zip(positive, negative)])
    x = np.concatenate([np.full(p+n, s) for p,n,s in zip(positive, negative, scores)])
    actual = SEARCH.weighted_metrics(scores, positive, negative)
    assert actual['roc_auc'] == pytest.approx(roc_auc_score(y, x), abs=1e-14)
    assert actual['average_precision'] == pytest.approx(average_precision_score(y, x), abs=1e-14)


@pytest.mark.parametrize('method', SEARCH.POOLING)
def test_compressed_segment_pooling_matches_existing_frame_reference(method):
    windows = [SimpleNamespace(start_frame=a, stop_frame=b)
               for a,b in [(0,5), (1,4), (2,6), (3,5)]]
    values = np.array([.15, .35, .7, .95])
    direct = aggregate_windows(windows, values, 6)[method]
    # One segment per frame here gives variable coverage from one to four.
    ids = np.full((6, 4), -1, dtype=int)
    for frame in range(6):
        active = [i for i,w in enumerate(windows) if w.start_frame <= frame < w.stop_frame]
        ids[frame, :len(active)] = active
    feature = np.zeros((4, 6))
    feature[:, 2] = values
    cache = dict(features=feature, ids=ids, counts=(ids>=0).sum(axis=1))
    actual = SEARCH.segment_scores(cache, dict(action_weight=0, pooling=method))
    np.testing.assert_allclose(actual, direct, rtol=0, atol=1e-15)


@pytest.mark.parametrize('key,value', [('action_temperature',float('nan')), ('action_temperature',float('inf')),
                                     ('action_temperature',0), ('action_image_weight',-.1)])
def test_invalid_recipe_rejected(key, value):
    with pytest.raises(ValueError):
        SEARCH.normalize_config({key:value})


def test_bank_free_recipe_has_no_memory_dependency():
    a = SEARCH.normalize_config(dict(action_weight=1, memory_image_weight=0, memory_scale='raw'))
    b = SEARCH.normalize_config(dict(action_weight=1, memory_image_weight=1, memory_scale='calibrated'))
    assert a == b
