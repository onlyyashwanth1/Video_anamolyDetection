"""Observation memory for the basic observe-then-detect model.

The captions of the observation windows (the first part of the same video)
become the normal "domain constitution": each distinct caption is one
MemoryEntry with its CLIP text embedding and how often it was seen.
"""

from collections import Counter
from typing import List

from anomaly_detection.datatypes import MemoryEntry
from anomaly_detection.memory.bank import MemoryBank


def build_domain_constitution(
    captions: List[str],
    min_occurrences: int = 1,
) -> List[str]:
    counts = Counter(captions)
    seen = []

    for caption in captions:
        if (
            caption not in seen
            and counts[caption] >= min_occurrences
        ):
            seen.append(caption)

    return seen


def embed_constitution(
    constitution: List[str],
    encoder,
    occurrence_counts: Counter = None,
) -> List[MemoryEntry]:
    entries = []

    for text in constitution:
        embedding = encoder.encode_text(text)
        count = (
            occurrence_counts[text]
            if occurrence_counts is not None
            else 1
        )
        entries.append(
            MemoryEntry(
                text=text,
                embedding=embedding,
                occurrence_count=count,
            )
        )

    return entries


def build_observation_memory(observation_events, encoder, min_occurrences=1):
    """Return (constitution captions, MemoryBank) built from captioned observation events."""
    captions = [event.description for event in observation_events]
    constitution = build_domain_constitution(captions, min_occurrences=min_occurrences)
    entries = embed_constitution(constitution, encoder, occurrence_counts=Counter(captions))
    return constitution, MemoryBank(initial_entries=entries)
