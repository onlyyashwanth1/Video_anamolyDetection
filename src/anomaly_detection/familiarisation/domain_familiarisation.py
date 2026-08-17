"""
Steps 5-6: Domain Familiarisation
------------------------------------------------------------------
Paper reference: "Phase 1: Online Domain Familiarisation" - during an
initial observation window, sampled events get described in plain
English, recurring patterns get written into a "Domain Constitution"
(the "Normal" notebook), and each sentence gets embedded via CLIP's
text encoder (Step 6) so it's directly comparable to image
fingerprints.

IMPLEMENTATION NOTE:
The paper calls for a full Multimodal LLM (LLaVA/GPT-4o) to freely
caption each event. This implementation uses a local multimodal LLM
through Ollama, currently configured to use:

    gemma4:31b-it-q8_0

If Ollama/Gemma fails, caption_event() falls back to CLIP zero-shot
classification using the candidate descriptions below.

Swapping the VLM model only requires changing OLLAMA_VLM_MODEL.
build_domain_constitution() and embed_constitution() do not need
to change.
"""

from collections import Counter
from typing import List

import io
import os

from PIL import Image

from anomaly_detection.utils.types import EventChunk, MemoryEntry


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Local multimodal LLM served through Ollama.
OLLAMA_VLM_MODEL = "gemma4:31b-it-q8_0"


# A starting vocabulary of plausible "normal" activities.
# Extend this list for your specific deployment domain
# (hallway, warehouse, factory, agricultural field, etc.).
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


# ------------------------------------------------------------------
# Step 5: Domain Familiarisation / Event Captioning
# ------------------------------------------------------------------

def caption_event(
    event: EventChunk,
    encoder,
    candidate_labels: List[str] = None,
    raw_frames: List = None,
) -> str:
    """
    Captures the main activity in an event using a local multimodal LLM
    through Ollama (Gemma 4 31B).

    If Ollama/Gemma fails or no raw frame is available, falls back to
    CLIP zero-shot classification using candidate_labels.

    Parameters
    ----------
    event:
        EventChunk containing frame indices and average embedding.

    encoder:
        CLIP-style encoder providing:
            encode_text()
            cosine_similarity()

    candidate_labels:
        Optional list of candidate descriptions for the CLIP fallback.

    raw_frames:
        List of raw frame objects. Each frame is expected to have:
            - image
            - frame_number

    Returns
    -------
    str
        A concise plain-English caption describing the event.
    """

    # --------------------------------------------------------------
    # Primary method: Gemma 4 31B through Ollama
    # --------------------------------------------------------------

    if raw_frames and len(event.frame_indices) > 0:
        try:
            import ollama

            # Use the middle frame as the representative event frame.
            mid_idx = event.frame_indices[len(event.frame_indices) // 2]

            target_frame = None

            # First try direct list indexing.
            if 0 <= mid_idx < len(raw_frames):
                target_frame = raw_frames[mid_idx]

            # Otherwise try matching by frame_number.
            if target_frame is None:
                target_frame = next(
                    (
                        frame
                        for frame in raw_frames
                        if frame.frame_number == mid_idx
                    ),
                    None,
                )

            if target_frame is not None:
                # --------------------------------------------------
                # Convert NumPy/OpenCV-style image to PIL
                # --------------------------------------------------
                pil_img = Image.fromarray(target_frame.image)

                # --------------------------------------------------
                # Encode image as JPEG bytes for Ollama
                # --------------------------------------------------
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                # --------------------------------------------------
                # Prompt Gemma
                # --------------------------------------------------
                prompt = (
                    "Describe the main action or activity occurring in "
                    "this video frame concisely in one short, "
                    "plain-English sentence. "
                    "Focus only on visible actions, people, objects, "
                    "and relevant activity. "
                    "Do not mention that you are analyzing an image. "
                    "Do not use bullet points."
                )

                response = ollama.generate(
                    model=OLLAMA_VLM_MODEL,
                    prompt=prompt,
                    images=[img_bytes],
                )

                # Ollama normally returns a dictionary containing
                # the generated text in "response".
                caption = (
                    response.get("response", "")
                    .strip()
                    .replace("\n", " ")
                    .lower()
                )

                if caption:
                    return caption

        except Exception as e:
            print(
                f"[Ollama VLM Note] Failed with "
                f"'{OLLAMA_VLM_MODEL}': {e}"
            )
            print("[Ollama VLM Note] Falling back to CLIP.")


    # --------------------------------------------------------------
    # Fallback: CLIP zero-shot classification
    # --------------------------------------------------------------

    if event.average_embedding is None:
        raise ValueError(
            "This EventChunk has no average_embedding set. "
            "Make sure it came from build_event_tree(), which "
            "computes this automatically."
        )

    labels = candidate_labels or DEFAULT_CANDIDATE_LABELS

    best_label = None
    best_score = float("-inf")

    for label in labels:
        label_embedding = encoder.encode_text(label)

        score = encoder.cosine_similarity(
            event.average_embedding,
            label_embedding,
        )

        if score > best_score:
            best_label = label
            best_score = score

    return best_label


# ------------------------------------------------------------------
# Step 5: Build Domain Constitution
# ------------------------------------------------------------------

def build_domain_constitution(
    captions: List[str],
    min_occurrences: int = 1,
) -> List[str]:
    """
    Turns a list of raw captions, one per familiarisation-window event,
    into a deduplicated list of recurring "normal" patterns.

    Parameters
    ----------
    captions:
        List of captions generated for observed events.

    min_occurrences:
        A caption must appear at least this many times to be included.

        Default = 1:
            Include every observed caption.

        For real deployments, increasing this value can prevent
        one-off events from becoming part of the "normal" constitution.

    Returns
    -------
    List[str]
        Deduplicated captions preserving first-seen order.
    """

    counts = Counter(captions)

    # Preserve first-seen order while applying occurrence threshold.
    seen = []

    for caption in captions:
        if (
            caption not in seen
            and counts[caption] >= min_occurrences
        ):
            seen.append(caption)

    return seen


# ------------------------------------------------------------------
# Step 6: Embed Domain Constitution
# ------------------------------------------------------------------

def embed_constitution(
    constitution: List[str],
    encoder,
    occurrence_counts: Counter = None,
) -> List[MemoryEntry]:
    """
    Converts each Domain Constitution sentence into a MemoryEntry.

    Each sentence is embedded using the CLIP text encoder so it can
    later be compared directly against image/event embeddings.

    Parameters
    ----------
    constitution:
        List of normal-event descriptions.

    encoder:
        CLIP-style encoder providing encode_text().

    occurrence_counts:
        Optional Counter containing how often each caption occurred.

    Returns
    -------
    List[MemoryEntry]
        Embedded memory entries ready for MemoryBank.
    """

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