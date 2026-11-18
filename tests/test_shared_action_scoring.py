"""Shared prompt probes and stable reductions, without loading CLIP."""

from types import SimpleNamespace

import numpy as np
import pytest

from anomaly_detection.inference.action_scoring import ActionScorer


def encoder():
    vectors = {
        'n1': [0.1, 0.5], 'n2': [0.3, 0.0],
        'a1': [0.2, 0.1], 'a2': [0.7, 0.2], 'a3': [0.4, 0.3],
        'text_n': [0.8, 0.1], 'text_a': [0.0, 0.9],
        'caption': [0.1, 1.0],
    }
    return SimpleNamespace(encode_text=lambda text: np.asarray(vectors[text], dtype=float))


BASE = {'normal_prompts': ['n1', 'n2'], 'anomalous_prompts': ['a1', 'a2', 'a3']}


def reduce_expected(values, method):
    values = np.asarray(values, dtype=float)
    if method == 'max': return max(values)
    if method == 'mean': return values.mean()
    if method == 'top2': return np.sort(values)[-2:].mean()
    return .1 * np.log(np.exp(values/.1).mean())


@pytest.mark.parametrize('method', ['max', 'mean', 'top2', 'logmeanexp'])
def test_reduction_matches_hand_computed_margin(method):
    scorer = ActionScorer(encoder(), {**BASE, 'image_prompt_reducer': method,
                                    'prompt_reduction_temperature': .1})
    result = scorer.score_vectors(np.array([[1., 0.]]))
    expected = reduce_expected([.2, .7, .4], method) - reduce_expected([.1, .3], method)
    assert result['action_image_margin'][0] == pytest.approx(expected, abs=1e-12)
    assert result['action_score'][0] == pytest.approx(1/(1+np.exp(-expected/.1)))


def test_separate_prompt_lists_and_temperatures_are_used():
    scorer = ActionScorer(encoder(), {**BASE, 'text_normal_prompts': ['text_n'],
        'text_anomalous_prompts': ['text_a'], 'image_temperature': .2,
        'text_temperature': .4, 'image_weight': .75})
    result = scorer.score_vectors(np.array([[1., 0.]]), np.array([[0., 1.]]))
    assert scorer.normal is scorer.image_normal and scorer.anomalous is scorer.image_anomalous
    assert result['action_image_margin'][0] == pytest.approx(.4)
    assert result['action_text_margin'][0] == pytest.approx(.8)
    expected = 1/(1+np.exp(-2.))
    assert result['action_image_score'][0] == pytest.approx(expected)
    assert result['action_text_score'][0] == pytest.approx(expected)
    assert result['action_score'][0] == pytest.approx(expected)


@pytest.mark.parametrize('method', ['max', 'mean', 'top2', 'logmeanexp'])
@pytest.mark.parametrize('transform', ['before_fusion', 'after_fusion'])
def test_batch_and_event_scores_agree_for_distinct_probes(method, transform):
    settings = {**BASE, 'text_normal_prompts': ['text_n'], 'text_anomalous_prompts': ['text_a'],
                'image_prompt_reducer': method, 'text_prompt_reducer': method,
                'image_temperature': .2, 'text_temperature': .3,
                'image_weight': .7, 'action_transform': transform}
    scorer = ActionScorer(encoder(), settings)
    image = np.array([.8, .2]); text = encoder().encode_text('caption')
    batch = scorer.score_vectors(image[None], text[None])
    event = scorer.score(SimpleNamespace(average_embedding=image, description='caption'))
    for key in batch:
        assert event[key] == pytest.approx(batch[key][0], abs=1e-12)
    if transform == 'after_fusion':
        logits = .7*event['action_image_margin']/.2 + .3*event['action_text_margin']/.3
        assert event['action_score'] == pytest.approx(1/(1+np.exp(-logits)))
    else:
        assert event['action_score'] == pytest.approx(.7*event['action_image_score']+.3*event['action_text_score'])


def test_no_caption_uses_image_probe_and_image_scale():
    scorer = ActionScorer(encoder(), {**BASE, 'image_temperature': .2, 'text_temperature': .8,
        'text_normal_prompts': ['text_n'], 'text_anomalous_prompts': ['text_a']})
    image=np.array([1.,0.])
    batch=scorer.score_vectors(image[None])
    event=scorer.score(SimpleNamespace(average_embedding=image, description=''))
    for key in event:
        assert event[key] == pytest.approx(batch[key][0])
    assert event['action_text_margin'] == event['action_image_margin']
    assert event['action_score'] == event['action_image_score'] == event['action_text_score']


def test_legacy_defaults_remain_max_and_before_fusion():
    scorer=ActionScorer(encoder(), BASE)
    result=scorer.score_vectors(np.array([[1.,0.]]),np.array([[0.,1.]]))
    assert result['action_image_margin'][0] == pytest.approx(.4)
    assert result['action_text_margin'][0] == pytest.approx(-.2)
    expected=.5/(1+np.exp(-4.))+.5/(1+np.exp(2.))
    assert result['action_score'][0] == pytest.approx(expected)


@pytest.mark.parametrize('key', ['image_temperature','text_temperature','prompt_reduction_temperature'])
@pytest.mark.parametrize('value', [float('nan'),float('inf'),0.,-.1])
def test_invalid_temperatures_are_rejected(key,value):
    with pytest.raises(ValueError,match=key):
        ActionScorer(encoder(), {**BASE,key:value})


@pytest.mark.parametrize('settings', [
    {'image_prompt_reducer':'sum'}, {'text_prompt_reducer':'sum'}, {'action_transform':'unknown'},
    {'image_normal_prompts':[]}, {'text_anomalous_prompts':['a1','a1']},
])
def test_invalid_probe_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        ActionScorer(encoder(),{**BASE,**settings})


def test_logmeanexp_stays_finite_for_extreme_similarities():
    scorer=ActionScorer(encoder(),{**BASE,'image_prompt_reducer':'logmeanexp',
                                 'prompt_reduction_temperature':1e-8})
    result=scorer.score_vectors(np.array([[1e6,0.],[-1e6,0.]]))
    assert all(np.isfinite(value).all() for value in result.values())
