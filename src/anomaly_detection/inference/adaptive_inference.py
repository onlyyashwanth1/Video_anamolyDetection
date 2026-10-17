"""
Steps 8-9: Adaptive Inference - Contrastive Probing + Soft Threshold Gating
Dual-Probe Engine: Computes both Image-to-Text and Text-to-Text similarity.
"""

import math
from anomaly_detection.utils.types import EventChunk, AnomalyResult


def _sigmoid_amplify(raw_score: float, center: float = 0.55, steepness: float = 25.0) -> float:
    """
    Applies a sigmoid function to transform a narrow score band into
    a well-spread [0, 1] range.

    Args:
        raw_score: The raw anomaly score (1 - similarity), expected ~0.45-0.65.
        center: The midpoint of the sigmoid; scores below this -> low anomaly,
                scores above -> high anomaly.
        steepness: How aggressively to spread scores; higher = sharper transition.
    """
    x = steepness * (raw_score - center)
    return 1.0 / (1.0 + math.exp(-x))


def score_event(event: EventChunk, memory_bank, threshold: float, encoder=None,
                top_k: int = 10) -> AnomalyResult:
    """
    Enhanced Dual-Probe Scoring with Top-K and Sigmoid Amplification:

    1. Uses top_k_match to get the average similarity of the K nearest memory
       entries (much more discriminative than best-match alone).
    2. Combines image-to-memory and text-to-memory probes with weighted fusion.
    3. Applies sigmoid amplification to spread scores from the natural narrow
       cosine band into a well-separated [0, 1] range.
    """
    if event.average_embedding is None:
        raise ValueError(
            "This EventChunk has no average_embedding set. Make sure it came "
            "from build_event_tree(), which computes this automatically."
        )

    # --- Image probe (top-K) ---
    img_entry, img_best_sim, img_topk_sim = memory_bank.top_k_match(
        event.average_embedding, k=top_k
    )
    # Use a weighted combination: mostly top-K average, with a bonus from best match
    img_raw = 1.0 - (0.6 * img_topk_sim + 0.4 * img_best_sim)

    # --- Text probe (top-K) ---
    txt_raw = 0.0
    txt_entry = None
    if encoder is not None and event.description:
        desc_embedding = encoder.encode_text(event.description)
        txt_entry, txt_best_sim, txt_topk_sim = memory_bank.top_k_match(
            desc_embedding, k=top_k
        )
        txt_raw = 1.0 - (0.6 * txt_topk_sim + 0.4 * txt_best_sim)

    # --- Fuse probes ---
    if txt_raw > 0:
        # Weighted combination: image probe is generally more informative
        fused_raw = 0.65 * img_raw + 0.35 * txt_raw
    else:
        fused_raw = img_raw

    # --- Sigmoid amplification to spread scores ---
    anomaly_score = _sigmoid_amplify(fused_raw, center=0.55, steepness=25.0)

    best_entry = img_entry if img_raw >= txt_raw else txt_entry
    is_anomaly = anomaly_score > threshold

    return AnomalyResult(
        event=event,
        anomaly_score=round(anomaly_score, 4),
        is_anomaly=is_anomaly,
        closest_match=best_entry.text if best_entry else None,
    )

