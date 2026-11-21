# Architectures

The report (`BTP_THESIS_VAD/`, Chapter 4) proposes **B + J** and compares it with two
baselines. Stage numbers below follow Chapter 4 and the "Memory + Judge" walkthrough
page (https://claude.ai/artifact/YUqrMTZtJ3iU8gGB5jaQW3, source in `docs/site/`).

| Model | Config | Role |
|---|---|---|
| B + J | `b_plus_j.yaml` | proposed model |
| B | `memory_b.yaml` | proposed model without the judge (ablation) |
| Action + memory | `action_default.yaml` | baseline (80/20) |
| Shared configuration | `shared_best.yaml` | baseline (one setting for UCF-Crime and XD-Violence) |
| Observe-then-detect | `observe_detect.yaml` | the starting idea, single videos only |

Configs: `config/base.yaml` <- `config/architectures/<name>.yaml` <- `config/datasets/<dataset>.yaml`
(later files override earlier ones; see `src/anomaly_detection/config.py`).

---

## Perception (stages 1-4, shared by all models)

Code: `src/anomaly_detection/perception/`.

1. **Frame sampling:** 1 frame per second.
2. **Multi-scale CLIP image encoding:** CLIP ViT-L/14 at full and half resolution,
   averaged. Frames are not masked.
3. **Sliding windows:** 4 s windows with a 1 s stride; a window's image embedding is
   the mean of its frames.
4. **Temporal captioning:** Gemma 4 31B (Ollama, temperature 0) captions up to 4 frames
   per window and summarizes them into one sentence.

Benchmark runs cache stages 1-4 per video (`evaluation/video_cache.py`,
`outputs/checkpoints/<dataset>_unmasked/`), so switching model costs only the scoring.

---

## Proposed model: B + J (`memory_novelty`)

No prompts. Code: `scoring/memory_novelty.py`, `scoring/judge.py`,
`perception/text_encoder.py`, `memory/bank.py`.

**Normal memory.** The windows of the normal training videos (captions, MPNet caption
vectors, CLIP image vectors) form the bank `outputs/memory/<dataset>_unmasked_v1.npz`.
Every memory window is also scored against the rest of the memory with its own video
left out; these scores calibrate the z-scores below and are cached next to the bank
(`<bank>.novelty_<encoder>_a1.json`).

5. **Sentence vectors and per-video centring:** each caption is embedded with MPNet;
   the video's mean caption vector is subtracted (removes what the scene is) and the
   result renormalized.
6. **Caption novelty:** 1 - cosine similarity to the most similar centred memory caption.
7. **Image novelty:** 1 - cosine similarity of the window's CLIP image vector to the most
   similar memory window.
   Both are robust z-scores against the memory's own leave-one-video-out scores:
   z = (x - median) / (IQR / 1.349). **B** = z_caption + 0.25 x z_image.
8. **Memory-aware judge (J):** retrieve the 8 most similar distinct memory captions
   (MPNet); Gemma rates "how likely is something abnormal" from 0 to 9 given those normal
   moments; the rating is the expected digit under Gemma's token probabilities. It is
   z-scored against ratings of 2000 seeded memory windows (each retrieving only captions
   that also occur in other memory videos). Ratings are cached by exact prompt in
   `outputs/judge_cache/`, so Gemma is asked only for new prompts.
9. **Fusion:** score = B + 1.0 x z_judge (`judge_weight` chosen from {0.25, 0.5, 1, 2}).
10. **Frame scores:** mean of the windows covering a frame, smoothed over
    min(36 s, video length / 4).
11. **Alert:** frames above `threshold_z` (3, not tuned). AUC/AP do not depend on it.

B is stages 1-7 and 10-11 without the judge (`judge_weight: 0`).
B and B + J are **offline**: centring uses the whole video's captions and smoothing
looks ahead.

---

## Baseline 1: action + memory (`action`, `action_default.yaml`)

Code: `scoring/action.py`.

```
score  = 0.80 x action + 0.20 x memory          (memory_weight 0.2)
action = 0.875 x image-prompt score + 0.125 x caption-prompt score
```

- Prompt score: CLIP similarity to 26 anomalous prompts minus 26 normal prompts, divided
  by a temperature (0.1), through a sigmoid. Prompts: `research/config/ucf_action_prompts_expanded.yaml`
  and Appendix A of the report.
- Memory score: distance from the window to the k = 10 nearest windows in the saved
  normal-memory bank, mapped to [0, 1] by a sigmoid fitted on held-out normal videos
  (the 20% of normal training videos not used in the bank).
- Frame scores: mean of covering windows. Alert threshold 0.5935.
- 0.80 is the best balance of a 0-1 weight sweep (step 0.05) on UCF-Crime and XD-Violence.
  `memory_weight: 0` gives prompts only and loads no bank.

The saved runs in `outputs/evaluations/<dataset>/action_default/` were made at the
earlier 0.95 / 0.05 weight. The fused score is linear in the saved window components,
so the 80/20 numbers are re-scored exactly from those runs by
`outputs/results/thesis_analyses_20261119/action_memory80_analyses.py`.

## Baseline 2: shared configuration (`action`, `shared_best.yaml`)

`memory_weight: 0`: CLIP prompts only, no bank. Separate image and caption prompt sets
that name benchmark categories, reduced with log-mean-exp; frame scores
0.5 x (mean + max) of the covering windows with 12 s smoothing. One setting is used for
both UCF-Crime and XD-Violence.

---

## Starting idea: observe-then-detect (`observe_detect`)

Watch a scene, remember what is normal, flag what is not.
Code: `scoring/observe_detect.py`, `memory/observation.py`. Single videos only
(`scripts/run_video.py`).

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

---

## Results

Frame-level metrics pooled over all test frames (`scripts/collect_results.py` ->
`outputs/results/architectures/results.md`). "eval" = the 80% evaluation partition of the
grouped validation/evaluation split of UCF-Crime and XD-Violence.

| Model | UCF AUC (all / eval) | XD AP (all / eval) | ShanghaiTech AUC / AP | Avenue AUC / AP |
|---|---|---|---|---|
| **B + J (proposed)** | 78.10 / 79.50 | 64.86 / 65.82 | 63.96 / 56.81 | 63.86 / 44.91 |
| B | 72.58 / 71.65 | 54.00 / 56.10 | 62.17 / 53.76 | 61.27 / 32.54 |
| Action + memory (80/20) | 78.09 / 81.24 | 60.73 / 62.03 | 61.75 / 54.51 | 66.38 / 47.64 |
| Shared configuration | 81.58 / 83.07 | 74.36 / 73.90 | 52.09 / 46.16 | 58.05 / 33.36 |

- The shared configuration is best on UCF-Crime / XD-Violence, but its prompts name the
  benchmark categories; outside them it is weakest (ShanghaiTech near chance).
- Action + memory is best on Avenue.
- B + J is the best prompt-free model everywhere and best on ShanghaiTech.
- B + J analyses (ablation, sensitivity, alerts): `outputs/results/thesis_analyses_20261119/bj_analyses.json`.

Prompts, weights and B's smoothing were selected using benchmark data (validation
partitions, ShanghaiTech).

### Check of the restructured B / B + J code

The `memory_novelty.py` / `judge.py` code was run on the memory-only study's saved inputs
(CLIP features, Gemma captions, MPNet embeddings, judge ratings); no model was run.

| | Study | Restructured code |
|---|---|---|
| UCF B, AUC all / validation / evaluation | 72.577 / 78.190 / 71.649 | 72.578 / 78.204 / 71.646 |
| XD B, AP all / validation / evaluation | 54.003 / 44.907 / 56.101 | 54.002 / 44.905 / 56.100 |
| UCF B + J, AUC all / validation / evaluation | 78.071 / 78.032 / 79.474 | 78.073 / 78.039 / 79.473 |
| XD B + J, validation AUC / AP | 88.719 / 62.421 | 88.719 / 62.421 |

Window-level B differs by at most 0.007 z (float32 here, float16 in the study).

## Reproducing the results

The caption/CLIP caches (`outputs/checkpoints/<dataset>_unmasked/`) and banks
(`outputs/memory/<dataset>_unmasked_v1.npz`) are in place, so a re-run only scores:

```bash
python scripts/evaluate.py --dataset ucf --architecture b_plus_j --gpus 0,1,2,3
python scripts/collect_results.py
```

Judge ratings come from `outputs/judge_cache/`; to load the memory-only study's ratings
into it, run `scripts/import_judge_ratings.py outputs/results/memory_only_search_20261107/llm_rag_*.json`.
Report figures and the all-models table regenerate with
`BTP_THESIS_VAD/scripts/make_figures.py` and `make_rerun_table.py`.
