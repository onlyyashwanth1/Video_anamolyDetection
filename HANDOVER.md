# Handover

The project is finished. This page is for whoever picks it up next from GitHub alone,
without access to the original server. Start with [README.md](README.md) for what the
models are and how to run them; this page covers what you need to get running and
what is not in the repository.

## What to read first

1. [BTP_THESIS_VAD/BTP_report.pdf](BTP_THESIS_VAD/BTP_report.pdf): the final report (75 pages).
   Chapter 4 describes the proposed model (B + J); Chapter 5 has the experiments.
2. [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md): each model, its config and its settings.
3. [docs/results/results.md](docs/results/results.md): every model on every dataset, with
   all metrics (a copy of `outputs/results/architectures/results.md`).
4. The walkthrough pages in [docs/site/](docs/site/) (open the `.html` files in a browser).

## State of the code

- Branch `main` is the final state; `restructure` is identical to it.
- `pytest tests/` passes (183 tests, CPU only, no model downloads).
- Branch `live-memory-40` holds a separate single-video live mode (observe the first 40%
  of a video, then score the rest). It is not merged into `main`; see `docs/LIVE_MEMORY40.md`
  on that branch.
- `research/` holds the frozen scripts behind earlier studies. They need the older package
  layout at commit `a3cb86e` (see [research/README.md](research/README.md)):
  `git worktree add ../vad_a3cb86e a3cb86e`.

## What you need to run it

**Hardware.** The runs used NVIDIA H200 GPUs. Gemma 4 31B (Q8) needs a large GPU, about 40 GB
of memory; CLIP and MPNet are small.

**Software.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/                       # check the install
```

Install [Ollama](https://ollama.com), then `ollama pull gemma4:31b-it-q8_0`. CLIP ViT-L/14 and
MPNet download from Hugging Face on first use.

**Paths to change.** Configs point at the original server. Edit these before running:

| File | Setting |
|---|---|
| `config/base.yaml` | `models_dir` (your Ollama models folder) |
| `config/datasets/ucf.yaml`, `xd.yaml`, `shanghaitech.yaml`, `avenue.yaml` | dataset folders and annotation files |

**Datasets** (not included; download them from their official sources):

| Dataset | Layout the configs expect |
|---|---|
| UCF-Crime | `Normal_Videos_event/`, `Testing_Video/`, `Temporal_Anomaly_Annotation_Five_FPS.txt` |
| XD-Violence | `Normal_Videos/`, `Testing_Video/`, `annotations.txt`, `train_split.txt`, `test_split.txt` |
| ShanghaiTech | convert the official release with `scripts/prepare_shanghaitech.py` |
| CUHK Avenue | convert the official release with `scripts/prepare_avenue.py` |

## What is not in the repository

Everything under `outputs/` (about 17 GB: caption and feature caches, normal-memory banks,
evaluation runs, figure data) stayed on the original server and is not in git. That means:

- The numbers in the report and in `docs/results/` can be read but not re-scored without
  re-running the evaluations. A full run captions every video with Gemma, which takes many
  GPU hours per dataset; captions are cached so each later model run is fast.
- `BTP_THESIS_VAD/scripts/make_figures.py` and `make_rerun_table.py` read `outputs/` and
  will not run until the evaluations exist again. The report itself compiles without them,
  because its figures are already in `BTP_THESIS_VAD/Figures/`.
- To get the saved results, ask the authors for `thesis_results_20261121.tgz` (311 MB).

## Building the report

`bash BTP_THESIS_VAD/build.sh` writes `BTP_report.pdf` with [Tectonic](https://tectonic-typesetting.github.io);
set `TECTONIC=/path/to/tectonic` if it is not on your PATH. For Overleaf, zip the
`BTP_THESIS_VAD/` folder, upload it and use pdfLaTeX.

## Known limitations

- B and B + J smooth over the whole video and centre captions per video, so they score
  complete videos only and are not real-time models.
- The prompts in `action_default` and `shared_best` were tuned on the benchmarks, and
  `shared_best`'s prompts name the benchmark categories. That is why `shared_best` is the strongest on
  UCF-Crime / XD-Violence and the weakest on ShanghaiTech / Avenue.
- The saved `action_default` runs were made at a 0.95 / 0.05 weight; the reported 80/20
  numbers re-score those runs' saved components (see README Notes).

## Ideas for future work

See Chapter 6 of the report.
