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


import os
from PIL import Image

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


def caption_event(event: EventChunk, encoder, candidate_labels: List[str] = None, raw_frames: List = None) -> str:
    """
    Captions an event chunk using Gemini 3.6 Flash (Multimodal LLM Vision) if
    GEMINI_API_KEY or GOOGLE_API_KEY is available, otherwise falls back to zero-shot CLIP classification.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and raw_frames and len(event.frame_indices) > 0:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            mid_idx = event.frame_indices[len(event.frame_indices) // 2]
            target_frame = (
                raw_frames[mid_idx] if (0 <= mid_idx < len(raw_frames))
                else next((f for f in raw_frames if f.frame_number == mid_idx), None)
            )
            if target_frame is not None:
                import time
                pil_img = Image.fromarray(target_frame.image)
                prompt = (
                    "Describe the main action or activity occurring in this video frame concisely "
                    "in one short, plain-English sentence."
                )
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=[pil_img, prompt],
                        )
                        caption = response.text.strip().replace("\n", " ").lower()
                        if caption:
                            return caption
                    except Exception as e:
                        if attempt < 2 and ("503" in str(e) or "429" in str(e)):
                            time.sleep(1.5)
                            continue
                        print(f"[Gemini VLM Note: {e}]")
                        break

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
