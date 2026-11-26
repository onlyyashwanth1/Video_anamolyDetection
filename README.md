# Training-Free Domain-Independent Video Anomaly Detection

B.Tech project, IIT Indore. The full report is
[docs/BTP_report.pdf](docs/BTP_report.pdf).

**The problem.** Given a video, find the moments where something abnormal happens
(a fight, an explosion, a road accident), without training a model on labelled
anomalies.

**The idea.** Learn what *normal* looks like from ordinary videos, then flag whatever
does not match it. Everything uses frozen, ready-made models and nothing is trained or
fine-tuned:

- **CLIP** (ViT-L/14) turns video frames into image vectors.
- **Gemma 4** (31B, run locally through Ollama) writes a one-sentence caption of every
  4-second window.
- **MPNet** turns those captions into sentence vectors.

**The proposed model, B + J (memory + judge).** Captions and image vectors from normal
training videos are saved as a *normal memory*. A new window is suspicious when:

- **B (memory score):** its caption and its image are far from anything in the memory.
- **J (judge):** Gemma, shown the most similar normal moments from memory, rates how
  likely the new window is to be abnormal.

The two are added into one anomaly score per frame.

## Results

The proposed model compared with the two baselines and an ablation, on four public
benchmarks. Higher is better; for AUC, 50 means chance.

| Model | UCF-Crime (AUC) | XD-Violence (AP) | ShanghaiTech (AUC) | Avenue (AUC) |
|---|---|---|---|---|
| **B + J (proposed)** | 78.10 | 64.86 | **63.96** | 63.86 |
| B alone (no judge) | 72.58 | 54.00 | 62.17 | 61.27 |
| Action + memory (baseline) | 78.09 | 60.73 | 61.75 | **66.38** |
| Shared configuration (baseline) | **81.58** | **74.36** | 52.09 | 58.05 |

The shared configuration uses text prompts that name the UCF-Crime and XD-Violence
categories, so it wins there and drops close to chance elsewhere. B + J uses no prompts
and is the most even across all four datasets. What the metrics mean, and the full
numbers: [docs/results/results.md](docs/results/results.md).

## See it step by step

Two interactive pages walk through the models on a simulated street clip (illustrative
values, not benchmark results). Open them in any browser, offline:

- [docs/site/memory_judge.html](docs/site/memory_judge.html): the proposed B + J model
- [docs/site/training_free_vad.html](docs/site/training_free_vad.html): an earlier version
  of the action + memory baseline (it still shows a 95 / 5 weighting and an object-masking
  step, both later changed)

They are also online at https://onlyyashwanth1.github.io/Video_anamolyDetection/.

## Project structure

```
Video_anamolyDetection/
├── README.md               this overview
├── config/                 settings files that choose the model and the dataset
│   ├── base.yaml             settings shared by every model
│   ├── architectures/        one file per model (b_plus_j.yaml, memory_b.yaml, ...)
│   └── datasets/             one file per dataset (where its videos are)
├── src/anomaly_detection/  the code, as a Python package
│   ├── perception/           reads the video, CLIP image vectors, Gemma captions
│   ├── memory/               builds, saves and loads the normal memory
│   ├── scoring/              the models: how each one turns a window into a score
│   ├── temporal/             turns window scores into one score per frame
│   ├── reasoning/            talks to Gemma (Ollama)
│   └── evaluation/           benchmark datasets, metrics, multi-GPU runs
├── scripts/                command-line tools: score one video, run a benchmark, ...
├── tests/                  automated tests (CPU only, no model downloads)
├── docs/
│   ├── BTP_report.pdf        the project report
│   ├── DEVELOPER_GUIDE.md    which file does what, settings, output files
│   ├── ARCHITECTURES.md      how each model computes its score
│   ├── results/              the final results table, explained
│   ├── site/                 the two walkthrough pages
│   └── diagrams/             editable architecture diagram (draw.io)
├── data/                   the datasets go here (not included)
└── outputs/                created when you run: caches, memories, results (not in git)
```

## Quick start

You need Python 3.9+, an NVIDIA GPU with about 40 GB of memory for Gemma, and
[Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
pip install -e .
ollama pull gemma4:31b-it-q8_0
pytest tests/                       # checks the install; runs on CPU
```

CLIP and MPNet download from Hugging Face the first time they are used.

**Score your own video** with the starting observe-then-detect idea, which needs no
dataset: it learns what is normal from the first 40% of the same video.

```bash
python scripts/run_video.py --video my_clip.mp4
```

It prints a score for every window and writes a text report and `.srt` subtitles to
`outputs/`.

**Run a benchmark** (after putting the dataset in `data/`, see the developer guide):

```bash
python scripts/evaluate.py --dataset ucf --architecture b_plus_j --gpus 0
```

## Going further

- Working on the code? Start with [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).
- How each model works, stage by stage: [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md)
  and Chapter 4 of the report.
- Branch `live-memory-40` adds a live single-video mode (observe the first 40% of a
  video, then raise alerts on the rest). It is described in `docs/LIVE_MEMORY40.md` on
  that branch.
