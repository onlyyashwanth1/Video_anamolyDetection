# UCF-Crime evaluation

The benchmark builds normal memory from every video in `Normal_Videos_event`,
then scores every event in every video in `Testing_Video`. It does not use the
demo's 40% familiarisation split. No test events are promoted into the bank.
Models remain pretrained; observing known-normal videos supplies a reference
memory. This differs from LAVAD's no-reference-data method.

## Persistent normal memory

The default artifact is `outputs/memory/ucf_crime_normal.npz`, relative to the
project root. It remains on disk for future runs and is never automatically
deleted or overwritten. It contains captions, normalized embeddings,
occurrence counts and JSON metadata describing the model/configuration, every
source video's path/size/modification time, and processing counts.

The artifact is written only after **all** normal videos finish successfully.
The `all` stage reuses an existing compatible bank; `evaluate` requires one.
`build-memory` refuses an existing destination: specify another artifact path
to deliberately build a new version. Model/configuration and source-manifest
changes are rejected when loading stale memory. Split checks compare video
identifiers and resolved paths; they cannot detect renamed copies of videos.

The NPZ is self-contained and can be read with `numpy.load(...,
allow_pickle=False)`. Its keys are `texts`, `counts`, `embeddings` and `metadata`.
It is ignored by Git but retained locally. Persistence does not provide backup
against disk loss. Old pickle banks from the previous evaluator are not reused,
because the feature pipeline and provenance requirements have changed.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `processing.py` | Shared masking, multi-scale CLIP, event extraction and captions |
| `evaluation/dataset.py` | Folder discovery, split checks, annotations and frame alignment |
| `evaluation/bank.py` | Full normal-folder observation and permanent memory artifacts |
| `evaluation/metrics.py` | Score/decision metrics and ROC/PR plots |
| `evaluation/runner.py` | Fixed-memory inference and per-video/result reports |
| `scripts/evaluate_ucf_crime.py` | CLI arguments and stage selection |

Models are reused across videos; tracking resets between videos. Benchmark mode
raises contextual errors on failed decoding, masking or captioning rather than
silently substituting a different model. One video's sampled frames are held in
memory at a time. Automatic resume and prediction caching are not implemented.

## Evaluation protocol

The supplied folders currently contain 800 normal-memory videos and 290 test
videos (150 normal, 140 anomalous). The annotations match those 290 filenames.
HVLMCLD describes a 300-video official test split, so results must be identified
as the supplied 290-video split until that discrepancy is resolved. Videos are
not silently skipped.

The primary metric follows LAVAD Section 4 and HVLMCLD Section 4.3: frame-level
ROC-AUC using continuous anomaly scores pooled across all test videos. It is
not an average of per-video AUC values. Additional reports include average
precision and precision/recall/F1/accuracy at `inference.anomaly_threshold`.
No threshold is selected using test labels. Single-class videos have undefined
AUC/AP, represented as JSON null. Metrics use the 0–1 scale.

Gemma may review threshold-flagged events. Its final binary decisions receive
separate decision metrics; they do not replace continuous scores for AUC.
Disable `evaluation.llm_review` for score-only evaluation. Annotation labels are
never passed to captioning, memory construction or anomaly scoring.

Scores cover **every original video frame**, including gaps between sampled
frames and the final tail. An event's score is held until the next event starts.
Annotation intervals are interpreted as one-based inclusive indices. The
annotation time base is explicit: `native` or a numeric FPS for annotations
indexed on a resampled video. A filename containing `Five_FPS` alone does not
establish its time base. Sampled videos were 30 FPS, and Abuse030's annotation
end (1360) exceeds its duration at 5 FPS (~258 frames); the configuration thus
uses native indices. Every interval is checked against the video length during
evaluation. This range check catches invalid time bases, but cannot prove that
the ground-truth boundaries are semantically correct.

## Results and model consistency

Each evaluation writes a new directory with `run.json` (provenance/status),
`summary.json`, `curves.npz`, `roc.png`, `precision_recall.png`, and per-video
NPZ predictions plus JSON event descriptions/decisions. Memory and annotation
SHA-256 hashes identify the artifacts used. Failed runs retain completed
outputs for diagnosis but are not resumable.

The demo and benchmark share the same event processor. Multi-scale image
features are averaged, not concatenated, to preserve CLIP's image/text space.
Events use normalized mean features. The experimental random temporal
transformer is not used: without trained alignment it would destroy the
similarity space. Its source file remains available for future work.

This evaluates the project's own detector. It does not reproduce LAVAD's
caption cleaning, temporal LLM scoring and score refinement, or HVLMCLD's
normal/anomaly dictionaries and hierarchical feature method.

Caching/resume, local evaluation verification and GPU execution instructions
are deferred at the user's request. No normal bank or benchmark measurements
are generated merely by adding this implementation.

### Annotation endpoints at EOF

Native-frame annotation endpoints that exceed the video length by at most two
frames are clipped to the last available frame, with a warning identifying the
video and original endpoint. The interval must start within the video. Larger
overruns and overruns in resampled annotation coordinates remain errors. The
source annotation file is not modified, and `run.json` records the native-frame
tolerance. A scan of the supplied 290-video split found five such overruns
(Arson011, Arson016, Explosion033, Fighting003, Shooting015); the affected video
lengths were also verified by decoding. This handles endpoint discrepancies; it
does not establish that every annotation is temporally correct.
