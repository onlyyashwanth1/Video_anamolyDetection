# How each model works

This page explains how each model turns a video into anomaly scores. Stage numbers
follow Chapter 4 of the report and the walkthrough page `docs/site/memory_judge.html`.
For which file implements what, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

| Model | Config (`config/architectures/`) | Role in the report |
|---|---|---|
| B + J | `b_plus_j.yaml` | proposed model |
| B | `memory_b.yaml` | proposed model without the judge (ablation) |
| Action + memory | `action_default.yaml` | baseline 1 |
| Shared configuration | `shared_best.yaml` | baseline 2 |
| Observe-then-detect | `observe_detect.yaml` | the starting idea; single videos only |

## Terms used below

- **Window:** a 4-second piece of video. Windows start every second, so they overlap.
  Every model scores windows; the window scores are then turned into frame scores.
- **Normal memory (memory bank):** the windows of a dataset's normal training videos,
  each stored as a caption, an MPNet caption vector and a CLIP image vector. It is
  built once per dataset and saved to a file.
- **Novelty:** how far a new window is from its closest match in memory
  (1 - cosine similarity). High novelty = nothing like it was seen in normal videos.
- **Robust z-score:** z = (x - median) / (IQR / 1.349), computed against the memory's own
  novelty values. It puts different scores on one scale: z = 3 means "much more unusual
  than normal windows usually are".

---

## Perception (stages 1-4, shared by all models)

1. **Frame sampling:** 1 frame per second.
2. **CLIP image vectors:** CLIP ViT-L/14 encodes each frame at full and half
   resolution, and the two vectors are averaged.
3. **Sliding windows:** 4 s windows with a 1 s stride. A window's image vector is the
   mean of its frames' vectors.
4. **Captions:** Gemma 4 31B (temperature 0) captions up to 4 frames of each window and
   summarises them into one sentence describing what happens.

In benchmark runs these four stages are cached per video, so trying another model only
re-runs the scoring.

---

## Proposed model: B + J

No text prompts are used anywhere.

5. **Caption vectors and per-video centring:** each caption becomes an MPNet sentence
   vector. The video's average caption vector is subtracted (this removes "what the scene
   is", leaving "what is happening now") and the result is renormalised.
6. **Caption novelty:** 1 - cosine similarity to the closest centred caption in memory.
7. **Image novelty:** 1 - cosine similarity of the window's CLIP image vector to the
   closest window in memory.
   Both are turned into robust z-scores. The reference values come from scoring every
   memory window against the rest of the memory, with its own video left out.
   **B = z_caption + 0.25 × z_image.**
8. **Memory-aware judge (J):** the 8 most similar distinct normal captions are retrieved
   from memory. Gemma is shown them together with the new caption and asked to rate from
   0 to 9 how likely it is that something abnormal is happening. The rating is the
   expected digit under Gemma's token probabilities, so it is a smooth value, not a
   whole number. It is z-scored against the ratings of 2000 normal memory windows.
9. **Fusion:** score = B + 1.0 × z_judge (the weight was chosen from {0.25, 0.5, 1, 2}).
10. **Frame scores:** each frame gets the mean of the windows covering it, then the
    scores are smoothed over min(36 s, video length / 4).
11. **Alert:** frames above z = 3 are flagged. This threshold was not tuned, and AUC and
    AP do not depend on it.

**B** is the same model without stages 8-9 (`judge_weight: 0`).

B and B + J are **offline** models: centring uses all of a video's captions and the
smoothing looks ahead, so they score a complete video, not a live stream.

---

## Baseline 1: action + memory (`action_default.yaml`)

```
score  = 0.80 × action + 0.20 × memory
action = 0.875 × image-prompt score + 0.125 × caption-prompt score
```

- **Prompt score:** CLIP similarity to 26 "anomalous" text prompts minus similarity to
  26 "normal" ones, divided by a temperature (0.1), passed through a sigmoid. The
  prompts are in the config file and in Appendix A of the report.
- **Memory score:** the distance from the window to its 10 nearest windows in the normal
  memory, mapped to 0-1 by a sigmoid fitted on the held-out 20% of normal videos.
- **Frame scores:** mean of the windows covering each frame. Alert threshold 0.5935.
- The 0.80 / 0.20 split is the best balance found by sweeping the weight from 0 to 1 in
  steps of 0.05 on UCF-Crime and XD-Violence. `memory_weight: 0` means prompts only,
  and then no memory is loaded.

## Baseline 2: shared configuration (`shared_best.yaml`)

Text prompts only, no memory (`memory_weight: 0`). There are separate prompt sets for
images and captions, and they name the benchmark anomaly categories. Prompt similarities
are combined with a log-mean-exp. Frame scores are 0.5 × (mean + max) of the covering
windows, smoothed over 12 s. A single setting serves both UCF-Crime and XD-Violence,
which is where the name comes from.

---

## Starting idea: observe-then-detect (`observe_detect.yaml`)

Watch a scene, remember what is normal there, then flag what is not. It needs no
training videos, because the memory comes from the same video. Single videos only
(`scripts/run_video.py`).

1. The windows of the first 40% of the video (`observation_fraction`) are captioned, and
   their distinct captions become the memory (as CLIP text vectors).
2. Each later window is compared with that memory:
   - image probe: the window's image vector against the memory caption vectors
   - text probe: the window's caption against the memory caption vectors
   - distance = 1 - (0.6 × mean of the top-10 similarities + 0.4 × best similarity)
   - score = sigmoid(25 × (0.65 × image distance + 0.35 × text distance - 0.55))
3. Windows above 0.5 are flagged. With `llm_review: true`, Gemma gives a verdict and an
   explanation for each flagged window and can clear the flag.
   `allow_anomaly_promotion` (off by default) lets captions that keep reappearing join
   the memory.

---

## Results

See [results/results.md](results/results.md).
