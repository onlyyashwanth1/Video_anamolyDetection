"""
Steps 5-6: Domain Familiarisation  [NOT YET IMPLEMENTED]
------------------------------------------------------------------
Paper reference: "Phase 1: Online Domain Familiarisation" - during an
initial observation window (e.g. first 10 minutes / 500 frames), a
Multimodal LLM (LLaVA / GPT-4o) describes sampled events in plain English.
Recurring patterns get written into a "Domain Constitution" - a short list
of what's normal for this specific location. Each sentence then gets
embedded via CLIP's text encoder (Step 6) so it's directly comparable to
image fingerprints.

Where this fits in the pipeline:
    segmentation (Step 3/4) -> familiarisation (THIS MODULE), only during
    the initial observation window. After that window ends, this module
    stops being called and memory/inference take over.

Planned interface:
    caption_event(event: EventChunk, encoder: ClipEncoder) -> str
        Samples a representative frame from the event and asks a
        multimodal LLM to describe it in plain English.

    build_domain_constitution(captions: list[str]) -> list[str]
        Clusters/deduplicates captions into a short list of recurring
        "normal" patterns (the Domain Constitution).

    embed_constitution(constitution: list[str], encoder: ClipEncoder) -> list[MemoryEntry]
        Converts each constitution sentence into a MemoryEntry using
        encoder.encode_text(), ready to be handed to memory/memory_bank.py.

TODO:
    - Pick and integrate a multimodal captioning API/model (e.g. an LLM
      with vision input, called via API or a local model like LLaVA)
    - Implement caption clustering/deduplication logic
    - Wire the observation-window length in from config/config.yaml
      (see: familiarisation.observation_window_frames)
"""

from typing import List

from anomaly_detection.utils.types import EventChunk, MemoryEntry


def caption_event(event: EventChunk, encoder) -> str:
    raise NotImplementedError(
        "Event captioning via a multimodal LLM is not implemented yet. "
        "See this module's docstring for the planned interface and TODOs."
    )


def build_domain_constitution(captions: List[str]) -> List[str]:
    raise NotImplementedError(
        "Domain Constitution building is not implemented yet. "
        "See this module's docstring for the planned interface and TODOs."
    )


def embed_constitution(constitution: List[str], encoder) -> List[MemoryEntry]:
    raise NotImplementedError(
        "Constitution embedding is not implemented yet. "
        "See this module's docstring for the planned interface and TODOs."
    )
