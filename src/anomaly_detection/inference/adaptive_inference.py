"""
Steps 8-9: Adaptive Inference - Contrastive Probing + Soft Threshold Gating
                                                    [NOT YET IMPLEMENTED]
------------------------------------------------------------------
Paper reference: "Adaptive Inference Engine" - "Contrastive Probing"
(cosine similarity between a live event's embedding and every entry in
the memory bank) followed by "Soft Threshold Gating" (deciding anomaly
vs. normal based on whether the best match clears a tuned threshold).

Where this fits in the pipeline:
    memory bank (Step 7) + segmentation (Step 3/4) -> inference (THIS
    MODULE) -> reasoning (Step 10) only runs on events flagged here.

Planned interface:
    def score_event(event: EventChunk, event_embedding: np.ndarray,
                     memory_bank: MemoryBank, threshold: float) -> AnomalyResult
        Finds the best match in memory_bank, computes anomaly_score =
        1 - best_similarity, and sets is_anomaly = anomaly_score > threshold.
        Does NOT fill in `.explanation` - that's reasoning/llm_reasoning.py's job.

TODO:
    - Implement score_event() using MemoryBank.best_match()
    - Read the default threshold from config/config.yaml
      (inference.anomaly_threshold)
    - Add threshold-tuning helper (sweep thresholds against a labeled
      dataset and report AUC-ROC / false-positive-rate, per the paper's
      evaluation methodology)
"""

import numpy as np

from anomaly_detection.utils.types import EventChunk, AnomalyResult


def score_event(event: EventChunk, event_embedding: np.ndarray,
                 memory_bank, threshold: float) -> AnomalyResult:
    raise NotImplementedError(
        "Adaptive inference (contrastive probing + threshold gating) is "
        "not implemented yet. See this module's docstring for the planned "
        "interface and TODOs. Requires memory/memory_bank.py to be "
        "implemented first."
    )
