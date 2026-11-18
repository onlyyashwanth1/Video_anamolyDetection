# Video Anomaly Detection — Training-Free, Domain-Independent Framework

> **Branch `live-memory-40`:** single-video live mode. The first 40% of a video
> builds its own normal memory, and the rest is tested against it, producing an SRT
> with captions and NORMAL/ANOMALY labels. Run `scripts/run_live_memory.py`; see
> [docs/LIVE_MEMORY40.md](docs/LIVE_MEMORY40.md).

The new [shared model configuration](config/shared_best.yaml) reaches **83.12%
UCF-Crime ROC-AUC** and **71.76% XD-Violence AP**, using identical model settings
on both datasets and no reference memory bank. These are offline benchmark
development results. See the [design, search evidence and reproduction guide](docs/SHARED_MODEL_SEARCH.md).

The default UCF configuration uses the expanded 26 normal / 26 anomalous
phrases, 95% action evidence and 5% visual normal memory. Inside the action
score, visual/caption evidence is weighted 87.5% / 12.5%. See the
[model and memory setup](docs/UCF_MODEL.md).
It uses overlapping windows and temporal captions as described in the
[window pipeline](docs/WINDOW_PIPELINE.md).

For several GPUs on one machine, see the [resumable multi-GPU evaluator](docs/MULTI_GPU.md).
For the quantized Ollama Gemma model on this server, see
[UCF-Crime in a detached screen](docs/UCF_OLLAMA_SCREEN.md).

A B.Tech project implementation of a training-free video anomaly detection
system: pretrained encoder weights are frozen. The current default uses a
saved normal-reference bank and development-selected prompts and fusion
weights. These are dataset adaptations even though neural weights are not
fine-tuned.

See `docs/PROJECT_STATUS.md` for exactly what's implemented vs. still a stub.

UCF-Crime benchmarking with a permanent normal memory bank is described in
[`docs/UCF_CRIME_EVALUATION.md`](docs/UCF_CRIME_EVALUATION.md). It uses separate
normal-memory and test folders. The fixed model's recorded ROC-AUC is **77.57%**
on all 290 supplied videos; see the [result summary](outputs/results/ucf_expanded95_summary.json).
Prompts and weights were selected during development using some of these test
videos, so this is a development result rather than an independent test estimate.
Current captioning uses local Gemma via Ollama. Demo and benchmark share
mean-pooled multi-scale CLIP windows.

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
│       │   └── object_masking.py   ← Step 4: YOLO-World + ByteTrack       [DONE]
│       ├── familiarisation/
│       │   └── domain_familiarisation.py  ← Steps 5-6: Domain Constitution [DONE]
│       ├── memory/
│       │   └── memory_bank.py      ← Step 7: Dynamic Textual Memory Bank  [DONE]
│       ├── inference/
│       │   └── adaptive_inference.py  ← Steps 8-9: Similarity + Threshold [DONE]
│       ├── reasoning/
│       │   └── llm_reasoning.py    ← Step 10: LLM Explanation             [DONE]
│       └── pipeline.py             ← orchestrates all of the above
├── scripts/
│   ├── run_pipeline.py             ← CLI (Steps 1-3 only)
│   └── run_full_pipeline.py        ← CLI (Full Steps 1-10)
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

## Run it (Full Pipeline, Steps 1-10)

```bash
python scripts/run_full_pipeline.py --video path/to/your/video.mp4
# or:
python scripts/run_full_pipeline.py --webcam
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
- **Captioning (Steps 5-6)** uses local Gemma through Ollama for frame captions and temporal summaries in the default configuration.
- **LLM verdict review (Step 10)** is disabled in the selected scoring configuration; optional review uses local Gemma.

Everything else follows the paper's design as closely as possible.

## Configuration

The [UCF model guide](docs/UCF_MODEL.md) describes the fixed scoring formula,
normal memory bank, and evaluation command.

Selected scoring settings live in `config/config.yaml`:
- `segmentation.coarse_threshold` / `fine_threshold` — how eagerly the
  system splits video into events
- `inference.anomaly_threshold` — alert cutoff; changing
  this cutoff does not change ROC-AUC
- `inference.action_weight` / `action.image_weight` / `memory_image_weight` —
  action-versus-memory, visual-versus-caption action, and visual-versus-text memory weights
- `familiarisation.observation_window_*` — how long the system watches
  before it starts alerting
- `memory.promotion_repeat_count` — how many times a new pattern must
  recur before being auto-added to "normal"
