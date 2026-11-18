"""Basic observe-then-detect architecture (the project's starting idea).

Watch the first observation_fraction of a video, caption it, and keep the
distinct captions as this scene's normal memory (memory/observation.py). Every
later window is compared with that memory:

    image probe: window CLIP image  vs memory caption CLIP text embeddings
    text probe:  window caption text vs memory caption CLIP text embeddings
    raw = 1 - (0.6 * mean of top-k similarities + 0.4 * best similarity), per probe
    fused = 0.65 * image raw + 0.35 * text raw  (image raw only, without a caption)
    score = sigmoid(25 * (fused - 0.55))

Flagged windows can be reviewed by Gemma (reasoning/explain.py) and repeatedly
unmatched captions can be promoted into memory (allow_anomaly_promotion).
This architecture runs on single videos only; it has no benchmark runner.
"""

import math

from anomaly_detection.datatypes import AnomalyResult, EventChunk


def _sigmoid_amplify(raw_score: float, center: float = 0.55, steepness: float = 25.0) -> float:
    """Spread the narrow cosine-distance band (~0.45-0.65) over [0, 1]."""
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - center)))


def score_event(event: EventChunk, memory_bank, threshold: float, encoder=None,
                top_k: int = 10) -> AnomalyResult:
    if event.average_embedding is None:
        raise ValueError("This EventChunk has no average_embedding set. Make sure it came "
                         "from build_event_tree() or build_sliding_windows().")
    img_entry, img_best_sim, img_topk_sim = memory_bank.top_k_match(event.average_embedding, k=top_k)
    img_raw = 1.0 - (0.6 * img_topk_sim + 0.4 * img_best_sim)

    txt_raw, txt_entry = 0.0, None
    if encoder is not None and event.description:
        txt_entry, txt_best_sim, txt_topk_sim = memory_bank.top_k_match(
            encoder.encode_text(event.description), k=top_k)
        txt_raw = 1.0 - (0.6 * txt_topk_sim + 0.4 * txt_best_sim)

    fused_raw = 0.65 * img_raw + 0.35 * txt_raw if txt_raw > 0 else img_raw
    anomaly_score = _sigmoid_amplify(fused_raw)
    components = {"memory_score": anomaly_score, "memory_raw": fused_raw,
                  "memory_image_raw": img_raw, "memory_text_raw": txt_raw,
                  "combined_score": anomaly_score}
    best_entry = img_entry if img_raw >= txt_raw else txt_entry
    return AnomalyResult(event=event, anomaly_score=anomaly_score, is_anomaly=anomaly_score > threshold,
                         closest_match=best_entry.text if best_entry else None,
                         component_scores=components)


def run_observe_detect(video, processor, settings, temporal_mode="sliding_windows", review=None):
    """Score one processed (uncaptioned) video against memory from its own beginning.

    Returns (constitution, memory_bank, results for the windows after the boundary).
    review(result, constitution) -> result is applied to flagged windows when given.
    """
    from anomaly_detection.memory.observation import build_observation_memory

    events, encoder = video.events, processor.encoder
    fraction = settings.get("observation_fraction", 0.4)
    if temporal_mode == "sliding_windows":
        from anomaly_detection.perception.windows import split_observation
        observed, live = split_observation(events, fraction, video.frame_count)
    else:
        if len(events) < 2:
            raise ValueError(f"Only found {len(events)} top-level event(s) - need at least 2.")
        split = min(max(1, math.ceil(len(events) * fraction)), len(events) - 1)
        observed, live = events[:split], events[split:]
    processor.caption_events(observed, video.frames)
    processor.caption_events(live, video.frames)
    constitution, memory_bank = build_observation_memory(
        observed, encoder, min_occurrences=settings.get("min_caption_occurrences", 1))
    threshold, top_k = float(settings["threshold"]), int(settings.get("top_k", 10))
    promote = settings.get("allow_anomaly_promotion", False)
    results = []
    for event in live:
        result = score_event(event, memory_bank, threshold, encoder=encoder, top_k=top_k)
        if result.is_anomaly:
            if review is not None:
                result = review(result, constitution)
            if result.is_anomaly and promote:
                memory_bank.log_unmatched(event.description, encoder.encode_text(event.description))
                memory_bank.try_promote(event.description, settings.get("promotion_repeat_count", 3))
        results.append(result)
    return constitution, memory_bank, results
