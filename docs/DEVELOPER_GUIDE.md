# Developer guide

For people working on the code. It covers which file does which part of the model, how
the settings work, how to set up the datasets, what the scripts do, and what every
output file contains. Read the [README](../README.md) first for the overview and
[ARCHITECTURES.md](ARCHITECTURES.md) for how each model computes its score.

Contents:
1. [How a video flows through the code](#1-how-a-video-flows-through-the-code)
2. [Code map](#2-code-map)
3. [Settings (config files)](#3-settings-config-files)
4. [Setting up the datasets](#4-setting-up-the-datasets)
5. [Scripts](#5-scripts)
6. [Output files](#6-output-files)
7. [Tests](#7-tests)
8. [Common tasks](#8-common-tasks)
9. [The report](#9-the-report)

---

## 1. How a video flows through the code

The stages of B + J (the proposed model), and where each one lives in
`src/anomaly_detection/`:

| Stage | What happens | File | Main function / class |
|---|---|---|---|
| 1 | Read the video, sample 1 frame per second | `perception/video.py` | `VideoStream` |
| 2 | CLIP image vector per frame (full + half resolution) | `perception/clip_encoder.py` | `MultiScaleClipEncoder` |
| 3 | Group frames into 4 s windows, 1 s apart | `perception/windows.py` | `build_sliding_windows` |
| 4 | Gemma caption per window | `perception/captioner.py` | `caption_windows` |
| 1-4 | Runs stages 1-4 for one video | `perception/processor.py` | `VideoEventProcessor.process` |
| - | Build / save / load the normal memory | `memory/store.py` | `build_bank`, `save_bank`, `load_bank` |
| 5 | MPNet caption vectors | `perception/text_encoder.py` | `SentenceEncoder` |
| 5-7 | Centring, caption + image novelty, z-scores, B | `scoring/memory_novelty.py` | `MemoryNoveltyScorer` |
| 8 | Retrieve normal captions, Gemma rating, z-score | `scoring/judge.py` | `MemoryJudge` |
| 9 | B + judge_weight × z_judge | `scoring/memory_novelty.py` | `MemoryNoveltyScorer.score_video` |
| 10 | Window scores → frame scores, smoothing | `temporal/projection.py` | `project_window_scores` |
| 11 | Threshold → alerts; AUC / AP | `evaluation/metrics.py` | `compute_metrics` |

The baselines reuse stages 1-4 and 10-11 and replace stages 5-9 with
`scoring/action.py` (`ActionMemoryScorer`). Observe-then-detect uses
`scoring/observe_detect.py` and `memory/observation.py`.

`scoring/__init__.py` picks the model from the config key `architecture`:

| `architecture:` | Models | Scorer |
|---|---|---|
| `memory_novelty` | B, B + J | `MemoryNoveltyScorer` (+ `MemoryJudge` when `judge_weight > 0`) |
| `action` | action + memory, shared configuration | `ActionMemoryScorer` |
| `observe_detect` | observe-then-detect | `run_observe_detect` |

---

## 2. Code map

```
src/anomaly_detection/
├── config.py            loads a config file and the files it extends, merges them
├── datatypes.py         small data classes: Frame, EncodedFrame, EventChunk (a window), AnomalyResult
├── pipeline.py          one video end to end (used by scripts/run_video.py)
├── perception/
│   ├── video.py           opens a file, a webcam or a public Google Drive link; samples frames
│   ├── clip_encoder.py    CLIP image and text vectors
│   ├── windows.py         sliding windows; split_observation for observe-then-detect
│   ├── events.py          older scene-change segmentation (temporal.mode: events), not used by the final models
│   ├── captioner.py       Gemma captions of each window's frames, then a one-sentence summary
│   ├── text_encoder.py    MPNet sentence vectors (and a version that reads precomputed ones)
│   └── processor.py       runs the perception stages for a video
├── memory/
│   ├── bank.py            MemoryBank: caption vectors + texts, nearest-match search
│   ├── visual_text.py     VisualTextMemoryBank: adds CLIP image vectors and distance calibration
│   ├── store.py           build the bank from normal videos, save/load it, check it matches the settings
│   └── observation.py     memory from the first part of a video (observe-then-detect)
├── scoring/
│   ├── __init__.py        chooses the scorer from the config
│   ├── memory_novelty.py  model B (and the fusion with J)
│   ├── judge.py           the memory-aware Gemma judge J, with a rating cache
│   ├── action.py          CLIP text-prompt scoring, optionally mixed with memory (both baselines)
│   └── observe_detect.py  observe-then-detect scoring
├── temporal/
│   └── projection.py      window scores → frame scores (mean / max pooling, smoothing)
├── reasoning/
│   ├── ollama_runtime.py  starts one Ollama server per GPU, identifies the installed model
│   ├── gemma.py           Gemma calls: captions and text generation (Ollama or Transformers)
│   └── explain.py         optional Gemma explanation of a flagged window
└── evaluation/
    ├── dataset.py         finds videos, reads annotation files, frame labels
    ├── xd_dataset.py      XD-Violence's own folder layout and annotation format
    ├── video_cache.py     saves stages 1-4 per video, so runs can resume and be reused
    ├── multi_gpu.py       splits videos across GPUs, one worker per GPU
    ├── runner.py          scores the test set and writes predictions and summary
    ├── metrics.py         ROC-AUC, AP, precision / recall at the threshold, curves
    └── logging_utils.py   log setup
```

---

## 3. Settings (config files)

A run combines three YAML files, each overriding the one before it:

```
config/base.yaml  →  config/architectures/<model>.yaml  →  config/datasets/<dataset>.yaml
```

An architecture file names its parent with `extends:` (`b_plus_j.yaml` extends
`memory_b.yaml`, which extends `base.yaml`). `scripts/evaluate.py` adds the dataset file
last. Relative paths are relative to the repository root.

**`base.yaml`, shared by every model**

| Section | Key settings |
|---|---|
| `video` | `target_fps: 1.0` (frames sampled per second) |
| `encoder` | `model_name`: the CLIP model |
| `temporal` | `window_seconds: 4`, `stride_seconds: 1` |
| `captioning` | `max_frames_per_window: 4` |
| `memory_bank` | `calibration_fraction: 0.2` (normal videos held out for calibration), `calibration_seed`, `top_k` |
| `reasoning` | `model: gemma4:31b-it-q8_0`, `temperature: 0`, `num_ctx`, `base_port` (first Ollama port); optional `models_dir` (default: `$OLLAMA_MODELS` or `~/.ollama/models`) |
| `evaluation` | `annotation_fps`, `llm_review` |

**Model files** (`config/architectures/`) set the `architecture` and its section:

- `memory_novelty:` `text_encoder` (MPNet), `centering_alpha`, `image_weight: 0.25`,
  `judge_weight` (0 = B, 1.0 = B + J), `threshold_z`, `judge:` (`memories: 8`,
  `calibration_windows: 2000`, `num_ctx`)
- `action:` `memory_weight`, `image_weight`, `temperature`, `threshold`, and the prompt lists
- `observe_detect:` `observation_fraction: 0.4`, `top_k`, `threshold`, `llm_review`
- `temporal:` (overrides) `score_pooling`, `smoothing_seconds`, `smoothing_divisor`

**Dataset files** (`config/datasets/`) say where the videos are and where to keep that
dataset's memory and cache:

| Key | Meaning |
|---|---|
| `normal_dir`, `test_dir`, `annotation_file` | the dataset (XD-Violence uses `dataset_root` instead) |
| `memory_bank_path` | where the normal memory is saved, e.g. `outputs/memory/ucf_unmasked_v1.npz` |
| `checkpoint_dir` | where the per-video cache of stages 1-4 goes |
| `reasoning.base_port` | a different Ollama port range per dataset, so two runs can share a machine |

The `_unmasked` in file names is a version tag. An earlier version blurred the
background around detected objects before encoding; that step was removed, and caches
from it are refused.

---

## 4. Setting up the datasets

Put each dataset under `data/` (or make `data` a link to wherever they are stored).
The expected layout:

```
data/
├── UCFCrime/
│   ├── Normal_Videos_event/      normal training videos
│   ├── Testing_Video/            test videos
│   └── Temporal_Anomaly_Annotation_Five_FPS.txt
├── XDViolence/
│   ├── Normal_Videos/  Testing_Video/
│   ├── annotations.txt  train_split.txt  test_split.txt
├── ShanghaiTech/                 made by scripts/prepare_shanghaitech.py
│   ├── Normal_Videos/  Testing_Video/  Temporal_Anomaly_Annotation.txt
└── Avenue/                       made by scripts/prepare_avenue.py
    ├── Normal_Videos/  Testing_Video/  Temporal_Anomaly_Annotation.txt
```

**Annotation files** give each test video's abnormal frame ranges:

- UCF-Crime, ShanghaiTech, Avenue: `<video> <category> <start> <end> [<start> <end>]`, one
  line per test video; `-1 -1` means "no second range". Example:
  `Abuse028_x264.mp4  Abuse  165  240  -1  -1`
- XD-Violence: `<video> <start> <end> [<start> <end> ...]` (zero-based, inclusive); normal
  test videos are not listed.

ShanghaiTech and Avenue are released as image frames, so convert them first:

```bash
python scripts/prepare_shanghaitech.py --raw <extracted official folder> --output data/ShanghaiTech
python scripts/prepare_avenue.py       --raw <extracted official folder> --output data/Avenue
```

Both rebuild lossless videos, check them pixel for pixel against the frames, and write
the annotation file.

Check a dataset without loading any model:

```bash
python scripts/evaluate.py --dataset ucf --architecture b_plus_j --gpus 0 --dry-run
```

---

## 5. Scripts

| Script | What it does |
|---|---|
| `scripts/run_video.py` | Scores one video (file, webcam, or public Google Drive link) with any model. Writes `outputs/pipeline_results.txt` and `.srt` subtitles (`--burn` burns them into a copy of the video). B + J and the action baseline need `--dataset` so they can load that dataset's normal memory. |
| `scripts/evaluate.py` | Runs a benchmark: caches stages 1-4 for every video, builds or loads the normal memory, scores the test set and writes the results. Multi-GPU (`--gpus 0,1,2,3`) and resumable: re-running skips cached videos. `--stage build-memory` only builds the memory; `--dry-run` only checks the dataset and cache. |
| `scripts/run_screen.sh` | `evaluate.py` in a detached GNU `screen` session, logging to `outputs/screen_runs/`. Takes the same arguments. |
| `scripts/collect_results.py` | Reads the latest finished run of every model × dataset in `outputs/evaluations/` and writes one table to `outputs/results/architectures/results.md` and `.json`. |
| `scripts/smoke_ollama.py` | Quick check that Ollama and Gemma work (`--models-only`), or a full run on two short UCF-Crime videos. |
| `scripts/import_judge_ratings.py` | Merges saved judge ratings (a JSON of prompt → rating) into the judge cache, so Gemma is not asked again. |
| `scripts/prepare_shanghaitech.py`, `prepare_avenue.py` | Convert those datasets to the layout above. |

---

## 6. Output files

Everything is written under `outputs/` (not in git). The folders appear as you run things.

```
outputs/
├── checkpoints/<dataset>_unmasked/   cache of stages 1-4, one .npz per video
├── memory/                           normal memories, one per dataset
├── judge_cache/                      Gemma judge ratings, reused across runs
├── evaluations/<dataset>/<model>/<time>/   one folder per benchmark run
├── results/architectures/            the collected results table
├── ollama_logs/                      Ollama server logs per run
├── screen_runs/                      logs of runs started with run_screen.sh
└── pipeline_results.txt / .srt       output of run_video.py
```

**`checkpoints/<dataset>_unmasked/<hash>.npz`**: one video's windows (start and end
frames), Gemma captions and CLIP window vectors. The file name is a hash of the video's
path, size and modification time and of the perception settings. A cached file is used
only for the same video with the same settings; changing a setting such as the window
length means the cache is rebuilt.

**`memory/<dataset>_unmasked_v1.npz`**: the normal memory.

| Array | Content |
|---|---|
| `texts`, `embeddings`, `counts` | distinct normal captions, their vectors, how often each occurred |
| `visual_embeddings` | CLIP vector of every normal memory window |
| `visual_sources`, `visual_starts`, `visual_text_indices` | the video and start frame of each window, and which caption it has |
| `metadata` | the settings it was built with, calibration values, the list of normal videos |

The `.novelty_<encoder>_a1.json` file next to it holds B's z-score reference values,
computed once and reused.

**`judge_cache/<model>_ctx<n>.json`**: prompt → judge rating. Ratings are deterministic
(temperature 0, fixed seed), so a prompt seen before is never sent to Gemma again.

**`evaluations/<dataset>/<model>/<time>/`**, one benchmark run:

| File | Content |
|---|---|
| `summary.json` | the results. `overall`: `roc_auc`, `average_precision`, frame counts, and `threshold_metrics` (precision, recall, F1, accuracy, TP / FP / TN / FN at the alert threshold). `per_video`: the same for each test video. Also the numbers of videos and memory entries. Metrics are fractions (0.781 = 78.1%). |
| `run.json` | how the run was made: the full merged config, checksums of the memory and annotation files, start / end times, `status` (`complete` when finished) |
| `predictions/<video>.json` | one test video: `fps`, `frame_count`, `category`, and `events`, one entry per window: `start_frame`, `stop_frame`, `caption`, `components` (the score's parts, e.g. `text_novelty`, `image_novelty`, judge rating), `score`, `final_anomaly` (flagged or not) |
| `curves.npz` | ROC curve (`fpr`, `tpr`) and precision-recall curve (`precision`, `recall`) with their thresholds |
| `roc.png`, `precision_recall.png` | plots of those curves |

---

## 7. Tests

```bash
pytest tests/
```

All tests run on the CPU with small fake inputs, without downloading any model, in
about 20 seconds. Each test file is named after the part it checks
(`test_frame_projection.py` → `temporal/projection.py`, `test_ollama_backend.py` →
`reasoning/ollama_runtime.py`, and so on). `test_architectures.py` checks that every
config file loads, and checks the scoring maths of B and B + J.

---

## 8. Common tasks

**Reproduce a result.**
```bash
python scripts/evaluate.py --dataset ucf --architecture b_plus_j --gpus 0,1,2,3
python scripts/collect_results.py
```
The first run on a dataset is slow because Gemma captions every video: many GPU hours for
UCF-Crime and XD-Violence. After that every model reuses the cache, and a run only does
the scoring.

**Change a setting.** Copy a file in `config/architectures/`, edit it, and run it with
`--config path/to/file.yaml` instead of `--architecture`. Results go to a new
timestamped folder, so older runs are kept.

**Add a dataset.** Arrange it as in [section 4](#4-setting-up-the-datasets), copy a
file in `config/datasets/` and change its paths, its `memory_bank_path` and
`checkpoint_dir`, and choose an unused `base_port`. Then add its name to `DATASETS` in
`scripts/evaluate.py` (and to `--dataset` in `scripts/run_video.py` if needed).

**Add a model.** Write a scorer in `scoring/` with the same interface as the others
(`score_video(events, frame_count, fps)` returns one score per window, and `unit(scores)`
maps scores to 0-1), register it in `scoring/__init__.py`, and add a config file.

**Use Gemma without Ollama.** Set `reasoning.backend: transformers` and the environment
variable `GEMMA_MODEL_PATH` to a local Gemma model folder.

---

## 9. The report

`BTP_THESIS_VAD/` holds the LaTeX source of the report. See
[its README](../BTP_THESIS_VAD/README.md) for how to build it and which script makes
which figure.
