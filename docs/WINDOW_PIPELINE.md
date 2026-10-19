# Window-based detection

The default configuration now uses 4-second windows starting every second,
cached frame captions followed by temporal summaries, and a combination of
normal-memory novelty and normal-versus-anomalous action similarity. Models
remain frozen. This is an experimental implementation, not a measured benchmark
improvement. Scores are heuristic rankings, not calibrated probabilities.

## Processing and scoring

Each sampled frame is encoded once. Windows reuse those features, have explicit
exclusive native-frame boundaries, and include the final video tail. Each
selected frame is captioned once per processing call; overlapping windows reuse
the caption. Gemma summarizes timestamped captions without receiving labels or
the anomaly score. The cache is in memory, per video, not a persistent resume cache.

Action prompts are embedded once per scorer. Both the window image embedding and
its caption embedding are compared against normal and anomalous prompt groups.
The best anomalous similarity minus the best normal similarity is mapped through
a sigmoid separately for each probe. Their mean is the action score. The default
combined score is 50% action score and 50% existing normal-memory score.
`action_weight`, action `temperature`, and `anomaly_threshold` are unvalidated
starting settings: select them using a separate development split, never test
labels. A common numerical range does not establish statistical calibration.

Evaluation averages overlapping continuous window scores on the original frame
grid, then applies the alert threshold. Per-window JSON includes all component
scores and raw margins. When optional LLM review is enabled, reviewed binary
window decisions are combined by majority vote (ties normal); AUC/AP still use
the continuous pre-review scores. No memory updates occur in benchmarks.

The demo assumes the initial observation interval is normal. It excludes windows
crossing that time boundary so observation and inference share no frames.
Automatic promotion of repeated anomaly alerts is disabled by default; its
optional legacy path now stores text embeddings in the text memory.

## Running

Rebuild memory because the descriptions and window boundaries have changed:

```bash
.venv/bin/python scripts/evaluate_ucf_crime.py --stage all
.venv/bin/python scripts/evaluate_xd_violence.py --stage all
```

These commands process the full configured datasets and can take substantial
GPU time. New default banks are `outputs/memory/ucf_crime_windows_v2.npz` and
`outputs/memory/xd_violence_windows_v2.npz`. Existing banks are preserved. Subsequent
runs can use `--stage evaluate`. Use `--memory-bank-path` for another artifact
when changing window or caption settings; incompatible artifacts are rejected.

## Controlled comparisons

Use separate YAML files with these settings and pass `--config`:

| Comparison | temporal.mode | captioning.mode | inference.scoring_mode |
|---|---|---|---|
| Old architecture | events | middle_frame | memory |
| Windows only | sliding_windows | middle_frame | memory |
| Windows + temporal captions | sliding_windows | frame_captions_then_summary | memory |
| Action score only | sliding_windows | frame_captions_then_summary | action |
| Combined | sliding_windows | frame_captions_then_summary | memory_plus_action |

The old architecture can load the original memory with `--memory-bank-path`.
New scores retain full precision instead of rounding to four decimals. Each
window/caption combination needs a compatible memory; changing just scoring
weights or prompts does not require rebuilding it. Compare on the same split.

## Limitations and validation

At 1 FPS the four-frame sequence can still miss fast actions. Temporal summaries
can be inaccurate; prompts cover only the listed behaviors. No motion backbone
or audio input has been added. More summaries increase model inference cost.
The processor still buffers a complete video; this is not a streaming rewrite.
Completed-window output requires future frames relative to the window start.

Unit tests cover short videos, tail coverage, overlap averaging, disjoint demo
splits, caption order/cache reuse, strict failures, action evidence for familiar
anomalies, and incompatible memory rejection. Real-model GPU benchmarks and
held-out calibration must still be run to establish accuracy and runtime.
