# Project Status — Paper Section → Code Mapping

This table is the single source of truth for "what does this file implement,
and is it done." Keep it updated as you build each module — it's what makes
handing this to another AI (or a teammate) painless.

| Paper Step | What it does | File | Status |
|---|---|---|---|
| 1. Live Input Stream | Reads video/webcam, samples frames | `src/anomaly_detection/ingestion/video_stream.py` | ✅ Done + tested (logic verified) |
| 2. Frozen Foundational Encoder | CLIP image/text embeddings | `src/anomaly_detection/encoding/clip_encoder.py` | ✅ Done (needs `torch`+internet to actually run) |
| 3. Automated Temporal Decomposition | Groups frames into event tree | `src/anomaly_detection/segmentation/temporal_decomposition.py` | ✅ Done + tested (2/2 tests passing) |
| 4. Spatial Object-Centric Masking | YOLO-World + ByteTrack background masking | `src/anomaly_detection/masking/object_masking.py` | 🔲 Stub only — raises `NotImplementedError` |
| 5-6. Domain Familiarisation | Captioning (via CLIP zero-shot, see note below) + Domain Constitution + text embedding | `src/anomaly_detection/familiarisation/domain_familiarisation.py` | ✅ Done + tested (3/3 tests passing) |
| 7. Dynamic Textual Memory Bank | Stores/updates "normal" entries, familiarity counter | `src/anomaly_detection/memory/memory_bank.py` | ✅ Done + tested (2/2 tests passing) |
| 8-9. Adaptive Inference | Contrastive probing (cosine sim) + threshold gating | `src/anomaly_detection/inference/adaptive_inference.py` | ✅ Done + tested (1/1 test passing) |
| 10. LLM Reasoning | Explanation generation for flagged events via Gemini 2.5 Flash | `src/anomaly_detection/reasoning/llm_reasoning.py` | ✅ Done + integrated |
| Orchestration | Wires modules together | `src/anomaly_detection/pipeline.py` | ✅ Steps 1-3 & 5-10 fully wired (`run_full_pipeline()`) |
| Shared types | Frame, EncodedFrame, EventChunk, MemoryEntry, AnomalyResult | `src/anomaly_detection/utils/types.py` | ✅ Done |
| Config | All tunable numbers in one place | `config/config.yaml` | ✅ Done (thresholds are starting guesses, not tuned) |
| CLI (Steps 1-3 only) | Run just ingestion/encoding/segmentation | `scripts/run_pipeline.py` | ✅ Done |
| CLI (full pipeline) | Run Steps 1-9 end to end, print notebook + anomaly results | `scripts/run_full_pipeline.py` | ✅ Done |
| Tests | Verify logic without needing model downloads | `tests/test_temporal_decomposition.py`, `tests/test_familiarisation_memory_inference.py` | ✅ 8/8 passing |

## Known simplifications (documented, not accidental)

- **Step 3 (temporal decomposition)**: uses a similarity-drop heuristic instead of a trained GEBD model (no pip-installable pretrained checkpoint exists).
- **Steps 5-6 (captioning)**: the paper specifies a full Multimodal LLM (LLaVA/GPT-4o) for free-form captioning. We use CLIP itself in zero-shot classification mode instead — comparing an event's fingerprint against a fixed candidate list of plain-English descriptions (see `DEFAULT_CANDIDATE_LABELS` in `domain_familiarisation.py`) and picking the closest one. This works and is fully tested, but captions are limited to that candidate list rather than being freely generated. Swapping in a real multimodal LLM later only requires rewriting `caption_event()` — nothing else changes.
- **Familiarisation window**: the paper describes a frame/time-based observation window (e.g. "first 10 minutes"). For short demo clips, we instead use `familiarisation.observation_fraction` in `config.yaml` (e.g. first 40% of detected events). Both approaches are valid; the fraction-based one is just more practical for short test videos.

## How to pick up where this left off

1. Read the docstring at the top of the stub file for the step you're building next (each one has a "Planned interface" and "TODO" section already written).
2. Implement the function/class signatures exactly as documented — other modules (and `pipeline.py`'s commented-out sketch) already expect those exact names and shapes.
3. Delete the `raise NotImplementedError(...)` line once implemented.
4. Add a test in `tests/` following the same "no external downloads needed" pattern as `test_temporal_decomposition.py`, where possible.
5. Update this table's Status column.
6. Uncomment/extend the relevant lines in `pipeline.py`'s sketch at the bottom of the file.

## Suggested build order

~~Familiarisation (5-6) → Memory Bank (7) → Adaptive Inference (8-9)~~ **DONE.**

Remaining: **Masking (4)** → **LLM Reasoning (10)**

We deliberately built 5-9 before 4, since 4 (masking) only improves accuracy on an
already-working detector, while 5-9 is what makes the system detect anomalies at
all. As of now, the project has a genuinely working end-to-end anomaly detector
(minus background masking and LLM-generated explanations) - run it with
`scripts/run_full_pipeline.py`.
