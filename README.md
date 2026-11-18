# Training-Free Video Anomaly Detection

Frozen foundation models only (CLIP ViT-L/14, Gemma 4 31B via Ollama, MPNet); no
weights are trained or fine-tuned. Three architectures share one perception
pipeline and are chosen with a config file:

| Architecture | Config | Normal reference | Score |
|---|---|---|---|
| **Observe-then-detect** (basic model) | `config/architectures/observe_detect.yaml` | first 40% of the same video | top-k similarity of each later window to the observed captions |
| **Action + memory** | `action_default.yaml` (thesis default), `shared_best.yaml` | saved normal-memory bank (optional) | 0.95 x CLIP action prompts + 0.05 x calibrated memory; `memory_weight: 0` = prompts only |
| **Memory novelty B / B+J** | `memory_b.yaml`, `b_plus_j.yaml` | saved normal-memory bank | B = caption + image novelty z-scores; B+J adds the memory-aware Gemma judge |

How each one works, its settings and its known results: [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md).

## Layout

```
config/
  base.yaml                 shared perception, memory-bank and Gemma settings
  architectures/            one file per model (extends base.yaml)
  datasets/                 ucf, xd, shanghaitech, avenue: paths, bank, checkpoints, Ollama port
src/anomaly_detection/
  perception/               video sampling, 4 s windows, CLIP (unmasked), Gemma captions, MPNet
  memory/                   observation memory, saved normal-memory bank (build/save/load)
  scoring/                  observe_detect.py, action.py, memory_novelty.py (B), judge.py (J)
  temporal/                 window scores -> frame scores (pooling, smoothing)
  reasoning/                Gemma via Ollama/Transformers, optional explanations
  evaluation/               datasets, resumable video cache, multi-GPU runner, metrics
  pipeline.py               single-video entry point
scripts/
  evaluate.py               benchmark evaluation (multi-GPU, resumable)
  run_screen.sh             evaluate.py detached in GNU screen
  run_video.py              score one video (any architecture), text report + SRT
  migrate_unmasked_cache.py convert old masked checkpoints without re-captioning
  import_judge_ratings.py   reuse saved B+J judge ratings
  prepare_shanghaitech.py, prepare_avenue.py, smoke_ollama.py
research/                   frozen experiment scripts behind the thesis numbers (see research/README.md)
tests/                      CPU tests, no model downloads
docs/                       ARCHITECTURES.md, archive/ (pre-restructure notes)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # optional; scripts and tests also add src/ to the path
```

Gemma runs through Ollama (`ollama` on PATH, model `gemma4:31b-it-q8_0` installed).
CLIP and MPNet load from the Hugging Face cache.

## Run

```bash
# One video
python scripts/run_video.py --video clip.mp4                                   # observe-then-detect
python scripts/run_video.py --video clip.mp4 --architecture b_plus_j --dataset ucf

# Benchmarks (captions and CLIP features are cached per video and shared by all architectures)
python scripts/evaluate.py --dataset ucf --architecture action_default --gpus 0,1,2,3 --dry-run
bash scripts/run_screen.sh --dataset xd --architecture b_plus_j --gpus 0,1,2,3
```

`evaluate.py` first captions and embeds every pending video (normal videos too
when the bank must be built), then builds or loads the bank, scores the test set
and writes `outputs/evaluations/<dataset>/<architecture>/<time>/summary.json`.
`observe_detect` is single-video only.

## Tests

```bash
pytest tests/
```

## Notes

- Object masking (YOLO-World + ByteTrack background blur) was removed: it hurt
  memory scoring and XD-Violence. Feature caches and banks are now versioned
  `window_caption_unmasked_v3`; old masked ones are refused. See
  `scripts/migrate_unmasked_cache.py` to reuse their Gemma captions.
- Prompts in `action_default` / `shared_best` were chosen during development on
  the benchmarks; `shared_best`'s prompts name the benchmark categories.
- B and B+J smooth over the whole video and center captions per video, so they
  are offline (complete-video) models.
