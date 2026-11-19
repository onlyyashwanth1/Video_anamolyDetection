# Architectures

All three share perception (`src/anomaly_detection/perception/`):

1. Sample 1 frame per second.
2. Group frames into 4 s windows with a 1 s stride.
3. Embed each frame with CLIP ViT-L/14 at full and half resolution (averaged); a
   window's image embedding is the mean of its frames. Frames are **not masked**.
4. Gemma 4 31B (Ollama, temperature 0) captions up to 4 frames per window and
   summarizes them into one sentence.

Benchmark runs cache steps 1-4 per video (`evaluation/video_cache.py`), so
switching architecture costs only the scoring.

Configs: `config/base.yaml` <- `config/architectures/<name>.yaml` <- `config/datasets/<dataset>.yaml`
(later files override earlier ones; see `src/anomaly_detection/config.py`).

---

## 1. Observe-then-detect (`observe_detect`)

The starting idea: watch a scene, remember what is normal, flag what is not.
Code: `scoring/observe_detect.py`, `memory/observation.py`.

1. The windows of the first `observation_fraction` (40%) of the video are captioned;
   their distinct captions form the normal memory (CLIP text embeddings).
2. Each later window is compared with that memory:
   - image probe: window image vs memory caption embeddings
   - text probe: window caption vs memory caption embeddings
   - raw distance = 1 - (0.6 x mean of top-10 similarities + 0.4 x best similarity)
   - score = sigmoid(25 x (0.65 x image + 0.35 x text - 0.55))
3. Windows above `threshold` (0.5) are flagged; with `llm_review`, Gemma gives a
   verdict and explanation and can clear the flag. `allow_anomaly_promotion`
   (off) lets repeatedly unmatched captions join memory.

Single videos only (`scripts/run_video.py`). The very first version used
`temporal.mode: events` and `captioning.mode: middle_frame`.

---

## 2. Action + memory (`action`)

Code: `scoring/action.py`.

```
score  = (1 - memory_weight) x action + memory_weight x memory
action = image_weight x image-prompt score + (1 - image_weight) x caption-prompt score
```

- Prompt score: CLIP similarity to anomalous prompts minus normal prompts, divided by
  a temperature, through a sigmoid.
- Memory score: distance from the window to the k=10 nearest windows in the saved
  normal-memory bank, mapped to [0, 1] by a sigmoid fitted on held-out normal videos.
- `memory_weight: 0` is prompts only; no bank is built or loaded.

| Config | memory_weight | Prompts | Frame scores |
|---|---|---|---|
| `action_default.yaml` (thesis default) | 0.05 | 26 normal / 26 anomalous, image 0.875 | mean of covering windows |
| `shared_best.yaml` | 0 | separate image/caption prompt sets naming benchmark categories, log-mean-exp | 0.5 x (mean + max), 12 s smoothing |

The 5% memory changes little: with masked features, action only scored 77.49 UCF AUC /
53.74 XD AP against 77.57 / 53.41 at 0.95/0.05.

---

## 3. Memory novelty B and B+J (`memory_novelty`)

No prompts. Code: `scoring/memory_novelty.py`, `scoring/judge.py`,
`perception/text_encoder.py`.

**B** for each window of a test video:

1. Caption novelty: MPNet caption embedding minus the video's mean caption embedding
   (removes what the scene is), renormalized; 1 - cosine similarity to the most similar
   centered normal-memory caption.
2. Image novelty: 1 - cosine similarity of the CLIP image embedding to the most similar
   normal-memory window.
3. Both are robust z-scores against the normal memory itself (each memory window scored
   with its own video left out): z = (x - median) / (IQR / 1.349).
4. B = z_caption + 0.25 x z_image.

**J** (when `judge_weight` > 0):

1. Retrieve the 8 most similar distinct normal-memory captions (MPNet).
2. Gemma rates "how likely is something abnormal" from 0 to 9 given those normal
   moments; the rating is the expected digit under Gemma's token probabilities.
3. z-score against ratings of 2000 seeded normal-memory windows (each retrieving only
   captions that also occur in other memory videos).
4. Score = B + 1.0 x z_judge.

Frames: mean of covering windows, smoothed over min(36 s, video length / 4).
`threshold_z` (3, not tuned) sets the alert level; AUC/AP do not depend on it.
B and B+J are **offline**: centering uses the whole video's captions and smoothing
looks ahead.

Caches: the memory's leave-one-video-out statistics are stored next to the bank
(`<bank>.novelty_<encoder>_a1.json`); judge ratings are stored by exact prompt in
`outputs/judge_cache/`. A rating is requested from Gemma only for a new prompt.

---

## Results (2026-11-19, restructured code, unmasked features)

Frame-level metrics pooled over all test frames (`scripts/collect_results.py` ->
`outputs/results/architectures/results.md`). "eval" = the 80% evaluation partition of the
grouped validation/evaluation split; Avenue was never used for any choice.

| Model | UCF AUC (all / eval) | XD AP (all / eval) | ShanghaiTech AUC / AP | Avenue AUC / AP (clean) |
|---|---|---|---|---|
| action_default (95/5) | 75.53 / 80.29 | 59.12 / 60.47 | 61.05 / 52.68 | 68.96 / 49.88 |
| shared_best | 81.58 / 83.07 | 74.36 / 73.90 | 52.09 / 46.16 | 58.05 / 33.36 |
| B | 72.58 / 71.65 | 54.00 / 56.10 | 62.17 / 53.76 | 61.27 / 32.54 |
| B+J | 78.10 / 79.50 | 64.86 / 65.82 | 63.96 / 56.81 | 63.86 / 44.91 |

- shared_best is best on UCF/XD, but its prompts name the benchmark categories; it is weakest
  outside them (ShanghaiTech near chance).
- action_default is best on Avenue (its action term alone: 69.2 AUC).
- B+J is the best prompt-free model everywhere and best on ShanghaiTech.
- Numbers measured earlier with object-masked image features (removed in the restructure) are
  in `BTP_THESIS_VAD_backup_masked_20261119.tgz` only.

Prompts, weights and B's smoothing were selected using benchmark data (validation
halves, ShanghaiTech); only Avenue was held out completely.

### Check of the restructured B / B+J code (2026-11-18)

The new `memory_novelty.py` / `judge.py` were run on the study's saved inputs
(unmasked CLIP features, Gemma captions, MPNet embeddings, judge ratings); no model was run.

| | Study | Restructured code |
|---|---|---|
| UCF B, AUC all / validation / evaluation | 72.577 / 78.190 / 71.649 | 72.578 / 78.204 / 71.646 |
| XD B, AP all / validation / evaluation | 54.003 / 44.907 / 56.101 | 54.002 / 44.905 / 56.100 |
| UCF B+J, AUC all / validation / evaluation | 78.071 / 78.032 / 79.474 | 78.073 / 78.039 / 79.473 |
| XD B+J, validation AUC / AP | 88.719 / 62.421 | 88.719 / 62.421 |

Window-level B differs by at most 0.007 z (float32 here, float16 in the study).
All 25,991 UCF judge prompts are identical to the study's. B+J above uses the study's
judge calibration: the restructured calibration excludes only captions that occur
solely in the query's own video (the study excluded by one recorded source video),
so 440 of its 2000 UCF prompts (233 for XD) are new and still need Gemma ratings.

## Re-running after the restructure

1. Convert the masked checkpoints (no Gemma): `scripts/migrate_unmasked_cache.py --dataset ucf`
   (`--compute-missing --gpu N` encodes the held-out calibration normals with CLIP).
2. `scripts/evaluate.py --dataset ucf --architecture action_default --gpus ...` builds
   `outputs/memory/ucf_unmasked_v1.npz` and evaluates. The other architectures reuse
   the same checkpoints and bank.
3. For B+J, `scripts/import_judge_ratings.py outputs/results/memory_only_search_20261107/llm_rag_*.json`
   puts the study's ratings into the judge cache, so Gemma is asked only for prompts it
   never rated (with retrieval on a GPU the test prompts are identical).
