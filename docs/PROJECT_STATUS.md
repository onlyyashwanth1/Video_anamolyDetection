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
| 5-6. Domain Familiarisation | Captioning + Domain Constitution + text embedding | `src/anomaly_detection/familiarisation/domain_familiarisation.py` | 🔲 Stub only |
| 7. Dynamic Textual Memory Bank | Stores/updates "normal" entries, familiarity counter | `src/anomaly_detection/memory/memory_bank.py` | 🔲 Stub only |
| 8-9. Adaptive Inference | Contrastive probing (cosine sim) + threshold gating | `src/anomaly_detection/inference/adaptive_inference.py` | 🔲 Stub only |
| 10. LLM Reasoning | Explanation generation for flagged events | `src/anomaly_detection/reasoning/llm_reasoning.py` | 🔲 Stub only |
| Orchestration | Wires modules together | `src/anomaly_detection/pipeline.py` | ✅ Steps 1-3 wired; 4-10 sketched as comments |
| Shared types | Frame, EncodedFrame, EventChunk, MemoryEntry, AnomalyResult | `src/anomaly_detection/utils/types.py` | ✅ Done |
| Config | All tunable numbers in one place | `config/config.yaml` | ✅ Done (thresholds are starting guesses, not tuned) |
| CLI | Run the pipeline from the command line | `scripts/run_pipeline.py` | ✅ Done (runs Steps 1-3) |
| Tests | Verify logic without needing model downloads | `tests/test_temporal_decomposition.py` | ✅ 2/2 passing |

## How to pick up where this left off

1. Read the docstring at the top of the stub file for the step you're building next (each one has a "Planned interface" and "TODO" section already written).
2. Implement the function/class signatures exactly as documented — other modules (and `pipeline.py`'s commented-out sketch) already expect those exact names and shapes.
3. Delete the `raise NotImplementedError(...)` line once implemented.
4. Add a test in `tests/` following the same "no external downloads needed" pattern as `test_temporal_decomposition.py`, where possible.
5. Update this table's Status column.
6. Uncomment/extend the relevant lines in `pipeline.py`'s sketch at the bottom of the file.

## Suggested build order

Masking (4) → Familiarisation (5-6) → Memory Bank (7) → Adaptive Inference (8-9) → LLM Reasoning (10)

This follows the same order data actually flows through the pipeline, so each step can be tested against real output from the previous one instead of mocked data.
