# Video Anomaly Detection — Training-Free, Domain-Independent Framework

A B.Tech project implementation of a training-free video anomaly detection
system: no model is ever fine-tuned. Frozen, pretrained AI models (CLIP,
and later YOLO-World/an LLM) are used as fixed tools, and the system learns
what's "normal" for a new location by watching it briefly, rather than
being trained on labeled data for that specific place.

See `docs/PROJECT_STATUS.md` for exactly what's implemented vs. still a stub.

## Folder structure

```
anomaly_detection/
├── README.md                      ← you are here
├── requirements.txt                ← pip dependencies
├── pyproject.toml                  ← makes the project pip-installable
├── config/
│   └── config.yaml                 ← every tunable number, in one place
├── src/
│   └── anomaly_detection/          ← the actual package (import as `anomaly_detection`)
│       ├── utils/
│       │   └── types.py            ← shared data types (Frame, EventChunk, etc.)
│       │                             every other module imports from here
│       ├── ingestion/
│       │   └── video_stream.py     ← Step 1: Live Input Stream            [DONE]
│       ├── encoding/
│       │   └── clip_encoder.py     ← Step 2: Frozen Foundational Encoder  [DONE]
│       ├── segmentation/
│       │   └── temporal_decomposition.py  ← Step 3: GEBD + HGTree        [DONE]
│       ├── masking/
│       │   └── object_masking.py   ← Step 4: YOLO-World + ByteTrack       [STUB]
│       ├── familiarisation/
│       │   └── domain_familiarisation.py  ← Steps 5-6: Domain Constitution [STUB]
│       ├── memory/
│       │   └── memory_bank.py      ← Step 7: Dynamic Textual Memory Bank  [STUB]
│       ├── inference/
│       │   └── adaptive_inference.py  ← Steps 8-9: Similarity + Threshold [STUB]
│       ├── reasoning/
│       │   └── llm_reasoning.py    ← Step 10: LLM Explanation             [STUB]
│       └── pipeline.py             ← orchestrates all of the above
├── scripts/
│   └── run_pipeline.py             ← CLI entry point
├── tests/
│   ├── conftest.py                 ← makes imports work without installing
│   └── test_temporal_decomposition.py  ← passes right now, no downloads needed
└── docs/
    └── PROJECT_STATUS.md           ← paper-section → file mapping + what's left
```

**Why this structure:** one folder per pipeline stage, matching the paper's
diagram one-to-one. Every module has a single, clearly-documented job. This
means:
- Anyone (or any AI) reading `docs/PROJECT_STATUS.md` instantly knows what's
  built and what isn't, without reading every file.
- New code for Step 4 only ever touches `masking/object_masking.py` — it
  never requires reopening or restructuring `video_stream.py` or
  `clip_encoder.py`, even months later.
- All shared data shapes live in one file (`utils/types.py`), so modules
  never disagree about what a "Frame" or "EventChunk" looks like.

## Setup

```bash
pip install -r requirements.txt
# optional but recommended - makes `import anomaly_detection` work anywhere:
pip install -e .
```

The first run downloads CLIP weights (~600MB) via `transformers`, so you need
internet access at least once.

## Run it (Steps 1–3, currently working)

```bash
python scripts/run_pipeline.py --video path/to/your/video.mp4
# or:
python scripts/run_pipeline.py --webcam
```

## Run tests (no downloads needed)

```bash
python tests/test_temporal_decomposition.py
# or, if you have pytest installed:
pytest tests/
```

## What's simplified vs. the paper (and why)

- **Temporal decomposition (Step 3)** uses a similarity-drop heuristic
  (comparing consecutive CLIP fingerprints) instead of a dedicated trained
  GEBD model. There's no single, drop-in pretrained GEBD package — real
  implementations are research code with fiddly setup. This heuristic is
  fully functional right now and can be swapped later without touching
  any other module (`build_event_tree()`'s signature only needs a list of
  `EncodedFrame`).

Everything else follows the paper's design as closely as possible — see
each stub module's docstring for the exact planned interface before you
implement it.

## Tuning

All thresholds live in `config/config.yaml`, not hardcoded in any `.py`
file:
- `segmentation.coarse_threshold` / `fine_threshold` — how eagerly the
  system splits video into events
- `inference.anomaly_threshold` — the cutoff for flagging an anomaly (not
  tuned yet — needs labeled data + an AUC-ROC sweep, see
  `docs/PROJECT_STATUS.md`)
- `familiarisation.observation_window_*` — how long the system watches
  before it starts alerting
- `memory.promotion_repeat_count` — how many times a new pattern must
  recur before being auto-added to "normal"
