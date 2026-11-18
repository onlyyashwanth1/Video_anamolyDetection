"""An action-only model needs neither reference data nor an observation prefix."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from anomaly_detection.inference.action_scoring import ActionScorer
from anomaly_detection.inference.adaptive_inference import score_event
from anomaly_detection.utils.types import EventChunk


NORMAL = np.asarray([1., 0.], dtype=np.float32)
ANOMALOUS = np.asarray([0., 1.], dtype=np.float32)


def event(start=0, anomalous=True):
    return EventChunk(start, start + 3, float(start), float(start + 4),
                      average_embedding=ANOMALOUS if anomalous else NORMAL,
                      description='fight' if anomalous else 'walking',
                      stop_frame=start + 4, frame_indices=[start])


def encoder():
    return SimpleNamespace(encode_text=lambda text: NORMAL if text == 'walking' else ANOMALOUS)


def settings():
    return dict(scoring_mode='action', anomaly_threshold=.5,
                action=dict(normal_prompts=['walking'], anomalous_prompts=['fight'],
                            image_weight=.75, temperature=.1))


class ForbiddenReference:
    def __getattribute__(self, name):
        if name.startswith('__'):
            return object.__getattribute__(self, name)
        raise AssertionError(f'Action-only scoring accessed reference attribute {name}')


@pytest.mark.parametrize('reference', [None, ForbiddenReference()])
def test_action_scoring_never_inspects_reference(reference):
    config = settings()
    scorer = ActionScorer(encoder(), config['action'])
    observed = event()
    result = score_event(observed, reference, .5, action_scorer=scorer, settings=config)
    assert result.is_anomaly
    assert result.anomaly_score == scorer.score(observed)['action_score']
    assert result.component_scores['combined_score'] == result.anomaly_score
    assert result.closest_match is None
    assert all(not k.startswith('memory') for k in result.component_scores)


def test_action_scoring_requires_scorer_without_touching_reference():
    with pytest.raises(ValueError, match='ActionScorer'):
        score_event(event(), ForbiddenReference(), .5, settings=settings())


@pytest.mark.parametrize('count', [1, 4])
def test_pipeline_scores_every_window_without_loading_or_building_memory(monkeypatch, count):
    import anomaly_detection.pipeline as pipeline
    import anomaly_detection.evaluation.bank as banks
    import anomaly_detection.reasoning.ollama_runtime as runtime
    import anomaly_detection.segmentation.sliding_windows as windows
    from anomaly_detection.memory.memory_bank import MemoryBank

    events = [event(i, anomalous=(i % 2 == 0)) for i in range(count)]
    processor = Mock()
    processor.encoder = encoder()
    raw_frames = [object()]
    processor.process.return_value = SimpleNamespace(events=events, frames=raw_frames,
                                                      frame_count=count + 3)
    monkeypatch.setattr(pipeline, 'VideoEventProcessor', Mock(return_value=processor))
    load_reference = Mock(side_effect=AssertionError('Reference file must not be read'))
    build_constitution = Mock(side_effect=AssertionError('No input-video memory may be built'))
    embed_constitution = Mock(side_effect=AssertionError('No input-video memory may be embedded'))
    familiarisation = Mock(side_effect=AssertionError('No observation prefix may be withheld'))
    reference_search = Mock(side_effect=AssertionError('No normal-memory retrieval may run'))
    monkeypatch.setattr(banks, 'load_bank', load_reference)
    monkeypatch.setattr(pipeline, 'build_domain_constitution', build_constitution)
    monkeypatch.setattr(pipeline, 'embed_constitution', embed_constitution)
    monkeypatch.setattr(windows, 'split_familiarisation', familiarisation)
    monkeypatch.setattr(MemoryBank, 'top_k_match', reference_search)
    monkeypatch.setattr(runtime, 'caption_session', lambda config: nullcontext())
    monkeypatch.setattr(runtime, 'resolve_model', Mock(side_effect=AssertionError('No bank preload')))
    review = Mock(side_effect=AssertionError('Configured verdict review is disabled'))
    monkeypatch.setattr(pipeline, 'explain_anomaly', review)

    config = dict(inference=settings(), memory=dict(representation='visual_text',
                  allow_anomaly_promotion=False), temporal=dict(mode='sliding_windows'),
                  familiarisation=dict(observation_fraction=.4, min_caption_occurrences=1),
                  evaluation=dict(memory_bank_path='/missing/reference.npz', llm_review=False))
    constitution, bank, results = pipeline.run_full_pipeline('video.mp4', config)

    assert constitution == [] and len(bank) == 0
    assert [r.event for r in results] == events
    assert [r.is_anomaly for r in results] == [i % 2 == 0 for i in range(count)]
    assert results[0].event.start_frame == 0  # includes the would-be observation period
    processor.process.assert_called_once_with('video.mp4', caption=False)
    processor.caption_events.assert_called_once_with(events, raw_frames)
    for forbidden in (load_reference, build_constitution, embed_constitution,
                      familiarisation, reference_search, review):
        forbidden.assert_not_called()
