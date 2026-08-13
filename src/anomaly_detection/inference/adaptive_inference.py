"""
Steps 8-9: Adaptive Inference - Contrastive Probing + Soft Threshold Gating
------------------------------------------------------------------
Paper reference: "Adaptive Inference Engine" - "Contrastive Probing"
(cosine similarity between a live event's embedding and every entry in
the memory bank) followed by "Soft Threshold Gating" (deciding anomaly
vs. normal based on whether the best match clears a tuned threshold).

Does NOT fill in AnomalyResult.explanation - that's reasoning/llm_reasoning.py's
job (Step 10), which only runs on events this module flags as anomalies.
"""

from anomaly_detection.utils.types import EventChunk, AnomalyResult


def score_event(event: EventChunk, memory_bank, threshold: float) -> AnomalyResult:
    """
    Compares event.average_embedding against everything in memory_bank,
    and decides anomaly vs. normal based on `threshold`.

    anomaly_score = 1 - best_similarity, so:
      - a close match (similarity near 1.0) -> anomaly_score near 0.0 (normal)
      - a poor match (similarity near 0.0)   -> anomaly_score near 1.0 (anomalous)

    threshold is compared against anomaly_score directly: if
    anomaly_score > threshold, it's flagged.
    """
    if event.average_embedding is None:
        raise ValueError(
            "This EventChunk has no average_embedding set. Make sure it came "
            "from build_event_tree(), which computes this automatically."
        )

    entry, similarity = memory_bank.best_match(event.average_embedding)
    anomaly_score = 1.0 - similarity
    is_anomaly = anomaly_score > threshold

    return AnomalyResult(
        event=event,
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=is_anomaly,
        closest_match=entry.text if entry else None,
    )
