"""Action with memory_weight 0 needs neither a memory bank nor an observation prefix."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from anomaly_detection.datatypes import EventChunk
from anomaly_detection.scoring import needs_bank
from anomaly_detection.scoring.action import ActionMemoryScorer, ActionScorer


NORMAL = np.asarray([1., 0.], dtype=np.float32)
ANOMALOUS = np.asarray([0., 1.], dtype=np.float32)


def event(start=0, anomalous=True):
    return EventChunk(start, start + 3, float(start), float(start + 4),
                      average_embedding=ANOMALOUS if anomalous else NORMAL,
                      description='fight' if anomalous else 'walking',
                      stop_frame=start + 4, frame_indices=[start])


def encoder():
    return SimpleNamespace(encode_text=lambda text: NORMAL if text == 'walking' else ANOMALOUS)


def settings(memory_weight=0.0):
    return dict(memory_weight=memory_weight, threshold=.5, normal_prompts=['walking'],
                anomalous_prompts=['fight'], image_weight=.75, temperature=.1)


class ForbiddenReference:
    def __getattribute__(self, name):
        if name.startswith('__'):
            return object.__getattribute__(self, name)
        raise AssertionError(f'Action-only scoring accessed reference attribute {name}')


@pytest.mark.parametrize('reference', [None, ForbiddenReference()])
def test_action_scoring_never_inspects_reference(reference):
    scorer = ActionMemoryScorer(settings(), encoder(), reference)
    observed = event()
    result = scorer.score_event(observed)
    assert result.is_anomaly
    assert result.anomaly_score == ActionScorer(encoder(), settings()).score(observed)['action_score']
    assert result.component_scores['combined_score'] == result.anomaly_score
    assert result.closest_match is None
    assert all(not k.startswith('memory') for k in result.component_scores)


def test_memory_weight_needs_a_bank():
    assert not needs_bank({'architecture': 'action', 'action': settings(0.0)})
    assert needs_bank({'architecture': 'action', 'action': settings(.05)})
    with pytest.raises(ValueError, match='bank'):
        ActionMemoryScorer(settings(.05), encoder(), None)


@pytest.mark.parametrize('weight', [-.1, 1.5, float('nan')])
def test_memory_weight_must_be_a_fraction(weight):
    with pytest.raises(ValueError, match='memory_weight'):
        ActionMemoryScorer(settings(weight), encoder(), None)


@pytest.mark.parametrize('count', [1, 4])
def test_pipeline_scores_every_window_without_loading_or_building_memory(monkeypatch, count):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.memory.store as store
    import anomaly_detection.reasoning.ollama_runtime as runtime
    from anomaly_detection.memory.bank import MemoryBank

    events = [event(i, anomalous=(i % 2 == 0)) for i in range(count)]
    processor = Mock()
    processor.encoder = encoder()
    raw_frames = [object()]
    processor.process.return_value = SimpleNamespace(events=events, frames=raw_frames,
                                                      frame_count=count + 3, fps=1.0)
    monkeypatch.setattr(pipeline, 'VideoEventProcessor', Mock(return_value=processor))
    load_reference = Mock(side_effect=AssertionError('Reference file must not be read'))
    observe = Mock(side_effect=AssertionError('No input-video memory may be built'))
    reference_search = Mock(side_effect=AssertionError('No normal-memory retrieval may run'))
    monkeypatch.setattr(store, 'load_bank', load_reference)
    monkeypatch.setattr(pipeline, 'run_observe_detect', observe)
    monkeypatch.setattr(MemoryBank, 'top_k_match', reference_search)
    monkeypatch.setattr(runtime, 'caption_session', lambda config: nullcontext())
    monkeypatch.setattr(runtime, 'resolve_model', Mock(side_effect=AssertionError('No bank preload')))
    review = Mock(side_effect=AssertionError('Configured verdict review is disabled'))
    monkeypatch.setattr(pipeline, 'explain_anomaly', review)

    config = dict(architecture='action', action=settings(), temporal=dict(mode='sliding_windows'),
                  evaluation=dict(memory_bank_path='/missing/reference.npz', llm_review=False))
    constitution, bank, results = pipeline.run_full_pipeline('video.mp4', config)

    assert constitution == [] and len(bank) == 0
    assert [r.event for r in results] == events
    assert [r.is_anomaly for r in results] == [i % 2 == 0 for i in range(count)]
    assert results[0].event.start_frame == 0  # includes the would-be observation period
    processor.process.assert_called_once_with('video.mp4', caption=False)
    processor.caption_events.assert_called_once_with(events, raw_frames)
    for forbidden in (load_reference, observe, reference_search, review):
        forbidden.assert_not_called()
