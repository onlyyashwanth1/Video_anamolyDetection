"""
Steps 5-6: Domain Familiarisation
------------------------------------------------------------------
Paper reference: "Phase 1: Online Domain Familiarisation" - during an
initial observation window, sampled events get described in plain
English, recurring patterns get written into a "Domain Constitution"
(the "Normal" notebook), and each sentence gets embedded via CLIP's text
encoder (Step 6) so it's directly comparable to image fingerprints.

IMPLEMENTATION NOTE (documented simplification, same pattern as Step 3's
GEBD stand-in): the paper calls for a full Multimodal LLM (LLaVA/GPT-4o)
to freely caption each event. That requires an external API or a large
local model. Instead, caption_event() below uses CLIP itself in
"zero-shot classification" mode: it compares the event's fingerprint
against a fixed list of candidate plain-English descriptions and picks
the closest one. This is a real, working captioning mechanism - just
limited to the candidate list you provide, rather than being able to
describe absolutely anything. Swapping in a true multimodal LLM later
only requires rewriting caption_event(); build_domain_constitution() and
embed_constitution() don't need to change.
"""

from collections import Counter
from typing import List

from anomaly_detection.utils.types import EventChunk, MemoryEntry


# A starting vocabulary of plausible "normal" activities. Extend this list
# for your specific deployment domain (a hallway, a warehouse, etc.) -
# the richer this list, the more meaningful the captions will be.
DEFAULT_CANDIDATE_LABELS = [
    "an empty room with no people",
    "a person walking through the scene",
    "a person standing still",
    "a person sitting down",
    "multiple people present in the scene",
    "a person entering the frame",
    "a person leaving the frame",
    "an object being moved or picked up",
    "no significant motion, a static scene",
    "a person facing the camera",
]


def caption_event(event: EventChunk, encoder, candidate_labels: List[str] = None) -> str:
    """
    Approximates captioning by finding which candidate description's CLIP
    text embedding is closest to the event's average image embedding.
    Requires event.average_embedding to already be set (done automatically
    by segmentation/temporal_decomposition.py's build_event_tree()).
    """
    if event.average_embedding is None:
        raise ValueError(
            "This EventChunk has no average_embedding set. Make sure it came "
            "from build_event_tree(), which computes this automatically."
        )

    labels = candidate_labels or DEFAULT_CANDIDATE_LABELS
    best_label, best_score = None, -1.0
    for label in labels:
        label_embedding = encoder.encode_text(label)
        score = encoder.cosine_similarity(event.average_embedding, label_embedding)
        if score > best_score:
            best_label, best_score = label, score

    return best_label


def build_domain_constitution(captions: List[str], min_occurrences: int = 1) -> List[str]:
    """
    Turns a list of raw captions (one per familiarisation-window event,
    with repeats) into a deduplicated list of recurring "normal" patterns.

    min_occurrences: a caption must appear at least this many times to be
    included. Default of 1 means "include everything seen at least once" -
    fine for short demo clips. Raise this for longer, real deployments so
    one-off captions don't get treated as permanently normal.
    """
    counts = Counter(captions)
    # Preserve first-seen order, but only keep captions meeting the threshold
    seen = []
    for caption in captions:
        if caption not in seen and counts[caption] >= min_occurrences:
            seen.append(caption)
    return seen


def embed_constitution(constitution: List[str], encoder, occurrence_counts: Counter = None) -> List[MemoryEntry]:
    """
    Converts each Domain Constitution sentence into a MemoryEntry (Step 6),
    ready to be handed to memory/memory_bank.py's MemoryBank.
    """
    entries = []
    for text in constitution:
        embedding = encoder.encode_text(text)
        count = occurrence_counts[text] if occurrence_counts else 1
        entries.append(MemoryEntry(text=text, embedding=embedding, occurrence_count=count))
    return entries
