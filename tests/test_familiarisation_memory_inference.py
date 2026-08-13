"""
Sanity tests for Steps 5-9 (familiarisation, memory bank, adaptive
inference) that do NOT require downloading CLIP - uses a small FakeEncoder
with hand-crafted embeddings instead, so the logic can be verified before
running against real video.

Run with:  python tests/test_familiarisation_memory_inference.py
       or:  pytest tests/
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.utils.types import Frame, EventChunk
from anomaly_detection.familiarisation.domain_familiarisation import (
    caption_event, build_domain_constitution, embed_constitution,
)
from anomaly_detection.memory.memory_bank import MemoryBank
from anomaly_detection.inference.adaptive_inference import score_event


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """Stands in for ClipEncoder: encode_text() looks up a fixed dictionary
    instead of calling a real model, so tests run instantly with no downloads."""

    def __init__(self, text_embeddings: dict):
        self._text_embeddings = text_embeddings

    def encode_text(self, text: str) -> np.ndarray:
        if text not in self._text_embeddings:
            raise KeyError(f"FakeEncoder has no embedding defined for: {text!r}")
        return self._text_embeddings[text]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


def make_event(embedding, start=0, end=0):
    return EventChunk(start_frame=start, end_frame=end, start_time=float(start),
                       end_time=float(end), average_embedding=unit(embedding))


def test_caption_event_picks_closest_label():
    encoder = FakeEncoder({
        "an empty room with no people": unit([1, 0]),
        "a person walking through the scene": unit([0, 1]),
    })
    event = make_event([0.95, 0.05])  # closer to "empty room"

    caption = caption_event(event, encoder, candidate_labels=list(encoder._text_embeddings.keys()))
    assert caption == "an empty room with no people", f"got: {caption}"
    print("PASSED: caption_event picked the correct closest label")


def test_build_domain_constitution_dedupes_and_filters():
    captions = ["empty room", "empty room", "person walking", "rare one-off event"]

    # min_occurrences=1: keep everything, deduplicated, first-seen order
    constitution = build_domain_constitution(captions, min_occurrences=1)
    assert constitution == ["empty room", "person walking", "rare one-off event"], constitution

    # min_occurrences=2: only keep things seen at least twice
    constitution_strict = build_domain_constitution(captions, min_occurrences=2)
    assert constitution_strict == ["empty room"], constitution_strict
    print("PASSED: build_domain_constitution dedupes and filters correctly")


def test_embed_constitution_produces_memory_entries():
    encoder = FakeEncoder({"empty room": unit([1, 0])})
    entries = embed_constitution(["empty room"], encoder, occurrence_counts={"empty room": 3})

    assert len(entries) == 1
    assert entries[0].text == "empty room"
    assert entries[0].occurrence_count == 3
    assert np.allclose(entries[0].embedding, unit([1, 0]))
    print("PASSED: embed_constitution produces correct MemoryEntry objects")


def test_memory_bank_best_match_and_empty_case():
    bank_empty = MemoryBank()
    entry, score = bank_empty.best_match(unit([1, 0]))
    assert entry is None and score == -1.0, "empty bank should report no match"

    from anomaly_detection.utils.types import MemoryEntry
    bank = MemoryBank([
        MemoryEntry(text="empty room", embedding=unit([1, 0])),
        MemoryEntry(text="person walking", embedding=unit([0, 1])),
    ])
    entry, score = bank.best_match(unit([0.9, 0.1]))
    assert entry.text == "empty room", f"expected empty room, got {entry.text}"
    assert score > 0.9
    print(f"PASSED: MemoryBank.best_match found '{entry.text}' with score {score:.3f}")


def test_memory_bank_promotion_requires_repeats():
    bank = MemoryBank()
    description = "cleaning cart passes through"
    embedding = unit([0.5, 0.5])

    # Only 2 occurrences so far - should NOT promote with threshold=3
    bank.log_unmatched(description, embedding)
    bank.log_unmatched(description, embedding)
    assert bank.try_promote(description, repeat_threshold=3) is False
    assert len(bank) == 0

    # Third occurrence - NOW it should promote
    bank.log_unmatched(description, embedding)
    assert bank.try_promote(description, repeat_threshold=3) is True
    assert len(bank) == 1
    assert bank.entries[0].text == description
    print("PASSED: MemoryBank only promotes after enough repeated occurrences")


def test_score_event_flags_anomaly_correctly():
    from anomaly_detection.utils.types import MemoryEntry
    bank = MemoryBank([MemoryEntry(text="empty room", embedding=unit([1, 0]))])

    normal_event = make_event([0.95, 0.05])   # very close to "empty room"
    weird_event = make_event([0, 1])           # completely different

    normal_result = score_event(normal_event, bank, threshold=0.5)
    weird_result = score_event(weird_event, bank, threshold=0.5)

    assert normal_result.is_anomaly is False, f"expected normal, got score {normal_result.anomaly_score}"
    assert weird_result.is_anomaly is True, f"expected anomaly, got score {weird_result.anomaly_score}"
    print(f"PASSED: score_event correctly flagged normal (score={normal_result.anomaly_score}) "
          f"vs anomaly (score={weird_result.anomaly_score})")


if __name__ == "__main__":
    test_caption_event_picks_closest_label()
    test_build_domain_constitution_dedupes_and_filters()
    test_embed_constitution_produces_memory_entries()
    test_memory_bank_best_match_and_empty_case()
    test_memory_bank_promotion_requires_repeats()
    test_score_event_flags_anomaly_correctly()
    print("\nAll tests passed.")
