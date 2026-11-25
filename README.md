# Training-Free Domain-Independent Video Anomaly Detection

Frozen foundation models only (CLIP ViT-L/14, Gemma 4 31B via Ollama, MPNet); no
weights are trained or fine-tuned. All models share one perception pipeline and
are chosen with a config file. The B.Tech project report is in `BTP_THESIS_VAD/`.

| Model | Role in the report | Config | Normal reference | Score |
|---|---|---|---|---|
| **B + J** (memory + judge) | **proposed model** | `config/architectures/b_plus_j.yaml` | saved normal-memory bank | caption + image novelty z-scores (B) plus the memory-aware Gemma judge (J) |
| Memory score B | ablation of the proposed model | `memory_b.yaml` | saved normal-memory bank | B alone, no judge |
| Action + memory | baseline | `action_default.yaml` | saved normal-memory bank | 0.80 x CLIP action prompts + 0.20 x calibrated memory |
| Shared configuration | baseline | `shared_best.yaml` | none | CLIP action prompts only, one setting for UCF-Crime and XD-Violence |
| Observe-then-detect | starting idea (single videos only) | `observe_detect.yaml` | first 40% of the same video | similarity of later windows to the observed captions |

How each one works, its settings and its results: [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md).

## Results

Pooled frame-level metrics, the same numbers as the report
(`outputs/results/architectures/results.md`, refresh with `scripts/collect_results.py`):

| Model | UCF-Crime AUC | XD-Violence AP | ShanghaiTech AUC | Avenue AUC |
|---|---|---|---|---|
| **B + J (proposed)** | 78.10 | 64.86 | **63.96** | 63.86 |
| Memory score B | 72.58 | 54.00 | 62.17 | 61.27 |
| Action + memory (80/20) | 78.09 | 60.73 | 61.75 | **66.38** |
| Shared configuration | **81.58** | **74.36** | 52.09 | 58.05 |

The shared configuration's prompts name the benchmark categories, so it is strongest on
UCF-Crime / XD-Violence and weakest outside them. B + J uses no prompts and is the most
balanced across the four datasets.

## Walkthrough pages

Two interactive step-by-step pages explain the models on a simulated street clip
(illustrative data, not benchmark results). Each is one self-contained HTML file in
`docs/site/`: download it and open it in any browser, offline. With GitHub Pages
enabled (Settings > Pages > branch `main`, folder `/docs`) they are also served at
https://onlyyashwanth1.github.io/Video_anamolyDetection/.

- **Memory + Judge (B + J, the proposed model):** `docs/site/memory_judge.html`
  (source: `docs/site/memory_judge_src/`, rebuilt with `build.py`).
- **Training-free VAD (the earlier action + memory pipeline):** `docs/site/training_free_vad.html`.
  It shows the design at that time: the 95 / 5 action / memory weight and an
  object-masking step that was later removed (see Notes).

## Layout

```
config/
  base.yaml                 shared perception, memory-bank and Gemma settings
  architectures/            one file per model (extends base.yaml)
  datasets/                 ucf, xd, shanghaitech, avenue: paths, bank, checkpoints, Ollama port
src/anomaly_detection/
  perception/               video sampling, 4 s windows, CLIP, Gemma captions, MPNet
  memory/                   observation memory, saved normal-memory bank (build/save/load)
  scoring/                  memory_novelty.py (B), judge.py (J), action.py, observe_detect.py
  temporal/                 window scores -> frame scores (pooling, smoothing)
  reasoning/                Gemma via Ollama/Transformers, optional explanations
  evaluation/               datasets, resumable video cache, multi-GPU runner, metrics
  pipeline.py               single-video entry point
scripts/
  evaluate.py               benchmark evaluation (multi-GPU, resumable)
  run_screen.sh             evaluate.py detached in GNU screen
  run_video.py              score one video (any model), text report + SRT
  collect_results.py        results table of all models x datasets
  import_judge_ratings.py   reuse saved B+J judge ratings
  prepare_shanghaitech.py, prepare_avenue.py, smoke_ollama.py
  migrate_unmasked_cache.py one-off conversion of the old masked caches (done; inputs removed)
research/                   frozen experiment scripts behind earlier studies (see research/README.md)
tests/                      CPU tests, no model downloads
docs/                       ARCHITECTURES.md, site/ (walkthrough page source), diagrams/, archive/
BTP_THESIS_VAD/             the report: LaTeX source, figures, build script (see its README.md)
outputs/                    results, caches, memory banks (not in git)
backup/                     zipped final report and results (not in git)
```

Datasets are read from `/home/atripathi/prudhvi/Dataset/` (paths in `config/datasets/`).

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
python scripts/run_video.py --video clip.mp4 --architecture b_plus_j --dataset ucf
python scripts/run_video.py --video clip.mp4                                   # observe-then-detect

# Benchmarks (captions and CLIP features are cached per video and shared by all models)
python scripts/evaluate.py --dataset ucf --architecture b_plus_j --gpus 0,1,2,3 --dry-run
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

`research/tests/` is skipped on this tree; it runs from the `a3cb86e` snapshot
(see `research/README.md`).

## Notes

- Frames are not masked. An earlier object-masking stage (YOLO-World + ByteTrack
  background blur) was removed because it hurt memory scoring and XD-Violence;
  feature caches and banks are versioned `window_caption_unmasked_v3` and old
  masked ones are refused.
- The saved `action_default` runs in `outputs/evaluations/` were made at the old
  0.95 / 0.05 weight. The config is now 0.80 / 0.20, the report's baseline; its
  numbers re-score those runs' saved window components exactly
  (`outputs/results/thesis_analyses_20261119/action_memory80_analyses.py`).
- Prompts in `action_default` / `shared_best` were chosen during development on
  the benchmarks; `shared_best`'s prompts name the benchmark categories.
- B and B+J smooth over the whole video and centre captions per video, so they
  are offline (complete-video) models.
