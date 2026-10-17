"""
Steps 5-6: Domain Familiarisation using local Gemma 4 31B (transformers)
"""

from collections import Counter
from typing import List
from PIL import Image

from anomaly_detection.utils.types import EventChunk, MemoryEntry


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


def caption_event(
    event: EventChunk,
    encoder,
    candidate_labels: List[str] = None,
    raw_frames: List = None,
    strict: bool = False,
) -> str:
    if raw_frames and len(event.frame_indices) > 0:
        try:
            mid_idx = event.frame_indices[len(event.frame_indices) // 2]
            target_frame = None

            if 0 <= mid_idx < len(raw_frames):
                target_frame = raw_frames[mid_idx]

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
                from anomaly_detection.reasoning import local_gemma
                pil_img = Image.fromarray(target_frame.image)
                prompt = (
                    "Describe the main action or activity occurring in "
                    "this video frame concisely in one short, "
                    "plain-English sentence. "
                    "Focus only on visible actions, people, objects, "
                    "and relevant activity. "
                    "Do not mention that you are analyzing an image. "
                    "Do not use bullet points."
                )

                caption = local_gemma.generate_caption(pil_img, prompt)
                caption = caption.strip().replace("\n", " ").lower()

                if caption:
                    return caption
                raise ValueError("Gemma returned an empty caption")

        except Exception as e:
            if strict:
                raise RuntimeError("Gemma event captioning failed") from e
            print(f"[Local Gemma VLM Note] Failed: {e}")
            print("[Local Gemma VLM Note] Falling back to CLIP.")

    if strict:
        raise ValueError("No representative frame available for Gemma captioning")

    if event.average_embedding is None:
        raise ValueError(
            "This EventChunk has no average_embedding set. "
            "Make sure it came from build_event_tree()."
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
