# One shared model for UCF-Crime and XD-Violence

The selected configuration is [`config/shared_best.yaml`](../config/shared_best.yaml).
It uses the same frozen encoder, caption model, prompts, temperatures, fusion,
window projection, smoothing and alert threshold on both datasets. It uses no
normal-reference bank, per-dataset calibration or dataset-specific scoring rule.
Only input paths, video metadata and ground-truth annotations differ.

The configuration was selected jointly for **UCF-Crime pooled native-frame
ROC-AUC** and **XD-Violence pooled native-frame average precision (AP)**.
Independent production replay evaluated all 290 UCF videos and all 800 XD videos.

| Full supplied benchmark | Original model | Shared selected model | Gain (percentage points) |
|---|---:|---:|---:|
| UCF-Crime ROC-AUC | 77.5695% | **83.1203%** | **+5.5508** |
| XD-Violence AP | 53.4073% | **71.7558%** | **+18.3485** |

These are **benchmark development results**, not independent test estimates.
Earlier UCF prompt development already used some of the supplied test videos;
both benchmarks and the evaluation subsets have also been inspected during this
conversation. Search selection in this run used fixed, separate source-video
validation groups. Holding those groups separate prevents sibling XD segments
from appearing in both partitions, but cannot undo earlier benchmark exposure.

| Fixed source-group partition | UCF ROC-AUC | XD AP |
|---|---:|---:|
| Validation: 58 UCF / 170 XD videos | 82.3928% | 69.9308% |
| Evaluation: 232 UCF / 630 XD videos | 83.2395% | 72.1490% |

An independent paired source-group bootstrap on the evaluation groups found a
UCF AUC gain of **3.248 points**, with descriptive 95% interval
**[-0.108, +6.882] points**; the UCF gain is less certain across videos.
XD AP gained **17.251 points**, with interval **[+12.910, +21.225] points**.
These 200-draw intervals are conditional on the selected models and omit
search uncertainty. See the [comparison report and plots](../outputs/results/shared_model_comparison_20261026/report.md).

Raw predictions, per-video metrics, source identities and common configuration
hashes are in [`outputs/shared_model_evaluation/20261026`](../outputs/shared_model_evaluation/20261026).
Both datasets have configuration SHA-256
`10504dcd946fa4d34513c495f65d780c411fd237285eefc878cf3d4db6b87378`
and model SHA-256
`5b7ad7ca65ed1ed156776f65f09e16d722a98bc09af89fd7b60b3f5fba4c783c`.
The original baseline outputs and default configurations remain available for comparison.

## Scoring design

Feature extraction remains CLIP ViT-L/14, sampled at 1 FPS, with four-second
windows at one-second stride. Existing spatial masking and mean-pooled
multi-scale image encoding are retained. Frame captions and temporal summaries
use the same frozen `gemma4:31b-it-q8_0` model on both datasets. Final verdict
review is disabled. Neither CLIP nor Gemma weights were trained during this search.

The two probes share one fixed definition across datasets:

- The image probe uses 12 normal and 16 anomalous semantic prompts, each prefixed
  with `a video of`. For each group, reduce CLIP similarities with log-mean-exp
  at temperature 0.05, then subtract the normal reduction from the anomalous
  reduction. Normalizing by the number of prompts avoids favoring the larger group.
- The caption probe retains the original 26 normal / 26 anomalous action phrases,
  with maximum similarity within each group and anomalous-minus-normal margin.

The exact prompt strings are embedded in the shared YAML. Let `mi` and `mt` be
the image and caption margins:

```text
image_evidence   = sigmoid(mi / 0.020)
caption_evidence = sigmoid(mt / 0.025)
window_score    = 0.8125 * image_evidence + 0.1875 * caption_evidence
frame_score     = 0.5 * mean(covering window scores)
                + 0.5 * max(covering window scores)
final_score     = centered 12-second boxcar(frame_score)
```

The boxcar uses nearest-value padding, with native-frame width
`max(1, round(12 * fps))`, increased by one if even. It preserves all original
frames and clip tails. **This is an offline model**: centered smoothing uses
future context. The reported result is not a causal streaming evaluation.

The alert threshold stays at `0.5935198772337466` in both datasets. It was not
optimized in this ROC-AUC/AP search; neither target metric depends on it.
The YAML is a ranking configuration, and its threshold should not be interpreted
as a validated production false-alarm policy. In particular, UCF alert precision
at this inherited threshold remains low despite improved ROC-AUC.

## What was tested

Several thousand recipes were evaluated in staged searches; stages overlap,
so the counts below should not be summed as unique configurations.

| Stage | Main alternatives | Evidence |
|---|---|---|
| Existing XD components | Image/caption weights, action/memory weights, calibrated/raw visual/text memory, temperatures and pooling | [XD search](../outputs/results/xd_config_search_20261026/search.json) |
| Joint bank-free scoring | Common sigmoid temperatures, fusion before/after sigmoid, eight poolers, label-free video normalization | [Joint search](../outputs/results/shared_config_search_20261026/search.json) |
| Shared prompts | Eight fixed prompt sets, max/top-two/mean/log-mean-exp similarity reduction, cross-probe prompt sets, fusion and pooling | [Full prompt search](../outputs/results/shared_prompt_search_full_validation_20261026/search.json) |
| Nonlinear evidence fusion | Agreement/confidence gates, geometric/harmonic fusion, extrema, independent probe temperatures | [Fusion search](../outputs/results/ucf_gating_shared_prompts_20261026/search.json) |
| Temporal design | Mean/max/top-k/quantile projection, nearest centers, linear interpolation, triangular overlap, 0–32-second smoothing | [Temporal search](../outputs/results/shared_temporal_search_20261026/search.json) |

The biggest gain came from changing the image prompt set and its similarity
reduction. Fusion calibration helped, and temporal smoothing added a smaller
further improvement. Clip-level normalization and the tested nonlinear gates
did not justify inclusion. Nearest-center, linear and triangular projection
trailed the selected overlap rule. Dataset-specific reference memory was
excluded from the final model to satisfy the identical-state requirement.

Local CPU inference encoded 192 fixed prompts and 108,305 unique captions.
The eight prompt definitions were fixed before examining their results. Saved
window image embeddings allowed rescoring without repeating video captioning.
GPU inference was unavailable locally, so denser frame sampling, changes to
spatial masking, larger encoders and new visual features were not measured on
the full benchmarks. Their improvement cannot be assumed.

Selection maximized the mean relative validation improvement in the two target
metrics, requiring neither validation objective to regress when feasible.
There is no unique setting that independently maximizes both objectives. The
selected common compromise was chosen before final production replay. The
[temporal report](../outputs/results/shared_temporal_search_20261026/report.md)
also records common configurations favoring each objective. Those alternatives
would still need to be used identically on both datasets.

## Reproduce on the saved features

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/evaluate_shared_model.py \
  --config config/shared_best.yaml \
  --feature-cache outputs/results/shared_prompt_features_20261026 \
  --output-dir outputs/shared_model_evaluation/reproduction
```

The evaluator defaults to the original completed UCF and XD runs, checks feature
and caption compatibility, uses the actual `ActionScorer`, and writes native-frame
predictions. Choose a fresh output directory; it refuses to overwrite results.
It requires the prepared local feature cache. Source runs can be supplied with
repeated `--source-run` arguments. Dataset selection controls input data only.

For a new video, the per-window scorer is available through
`scripts/run_full_pipeline.py --config config/shared_best.yaml --video VIDEO`.
That interface prints window evidence; reported benchmark metrics additionally
use native-frame projection and smoothing in `frame_projection.py`.
The legacy memory-building evaluators are intended for the original bank-based
configurations; use `evaluate_shared_model.py` for this saved-feature replay.

## Verification

- Reconstructed original per-frame scores and pooled metrics before searching.
- Verified weighted segment AUC/AP against independently expanded native frames,
  including tied scores and mixed-label segments.
- Compared cached prompt margins with the production scorer across both datasets.
- Independently verified scalar/batch action scoring, separate prompt reductions,
  memory-free inference and whole-video coverage.
- Replayed the final YAML through production scoring on all 1,090 videos; resulting
  metrics match the independent search implementation to numerical precision.
- Verified identical model/configuration hashes for UCF and XD.
- Final repository test suite: **154 passed, 1 GPU-dependent test skipped**.

A fresh external split or untouched dataset is still required to measure
independent generalization. Changes to sampling or masking require fresh visual
inference and are not represented by these cached-feature results.
