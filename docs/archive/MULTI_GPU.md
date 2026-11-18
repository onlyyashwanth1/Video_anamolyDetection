# Single-machine multi-GPU evaluation

For quantized Gemma through Ollama and execution that survives an SSH disconnect,
see [UCF-Crime with Ollama in screen](UCF_OLLAMA_SCREEN.md).

`scripts/evaluate_ucf_multi_gpu.py` uses one independent model replica per GPU.
The original `evaluate_ucf_crime.py` and its defaults are unchanged. A running
serial evaluation cannot join this queue or export its unfinished in-memory
work; leave it running if you want to preserve that work.

## Run

On a Linux machine with six available GPUs, from the repository directory:

```bash
# Inspect the dataset and longest-first queue without loading models or using GPUs.
.venv/bin/python scripts/evaluate_ucf_multi_gpu.py \
  --gpus 0,1,2,3,4,5 --stage all --dry-run

# Start the actual job (for five GPUs use 0,1,2,3,4).
.venv/bin/python -u scripts/evaluate_ucf_multi_gpu.py \
  --gpus 0,1,2,3,4,5 --stage all \
  --checkpoint-dir outputs/checkpoints/ucf_multi_gpu \
  --memory-bank-path outputs/memory/ucf_crime_multi_gpu.npz
```

Run the second command inside your screen session. Only select GPUs available
to this job. No real-model multi-GPU throughput claim has been validated yet.
The dry run does not verify CUDA availability or whether models fit.
The model paths, dataset paths, and installed dependencies must be available
on the new machine, just as they are for the serial evaluator.

### XD-Violence

Use the XD entry point with the prepared dataset root:

```bash
.venv/bin/python scripts/evaluate_xd_multi_gpu.py \
  --gpus 0,1,2,3,4,5 \
  --dataset-root /home/nverma/training_free_vad/Datasets/XDViolence \
  --stage all --dry-run

.venv/bin/python -u scripts/evaluate_xd_multi_gpu.py \
  --gpus 0,1,2,3,4,5 \
  --dataset-root /home/nverma/training_free_vad/Datasets/XDViolence \
  --stage all
```

The root must contain `Normal_Videos/`, `Testing_Video/`, `train_split.txt`,
`test_split.txt`, and `annotations.txt`. The existing strict XD loader validates
split membership, disjointness, and annotation boundaries before any workers
start. It converts zero-based inclusive native-frame intervals to the shared
one-based inclusive representation. Normal test videos receive empty intervals.
As in the serial XD command, annotation FPS is native and optional LLM review
is disabled; frame/window captioning remains enabled. The final summary includes
pooled frame-level average precision and ROC-AUC.

XD defaults are isolated from UCF:

- Checkpoints: `outputs/checkpoints/xd_violence_multi_gpu`
- Memory: `outputs/memory/xd_violence_multi_gpu.npz`
- Results: `outputs/xd_violence_multi_gpu_evaluation/<timestamp>`

The stage, resume, GPU selection, and resource limitations below apply equally
to both datasets. Select XD inputs using `--dataset-root`; individual UCF-style
folder/annotation overrides are rejected.

GPU IDs are indices into the current `CUDA_VISIBLE_DEVICES` allocation when
that variable is set. For example, with `CUDA_VISIBLE_DEVICES=3,5`, pass
`--gpus 0,1`. Each fresh spawned worker sees only its assigned physical GPU,
as `cuda:0`; it cannot spread Gemma across the other workers' GPUs. Missing
CUDA or Gemma CPU/disk offload fails explicitly. Each GPU must fit a complete
Gemma/CLIP/detector replica.

## Processing and resume

The shared queue starts the longest videos first and assigns the next video
to whichever worker becomes free. Workers retain their models between videos.
Frame encoding, detection, frame captioning, and temporal summaries for both
normal and test videos run in parallel. Sampling, window boundaries, caption
prompts, and scoring settings are unchanged. Whole videos stay on one worker,
so tracker state and overlapping windows are not split across devices.

Each completed video atomically saves captions, window embeddings, frame count,
and FPS to a compressed NPZ. Raw video frames are not saved. Cache identity
includes the model specification and source path, size, and modification time.
Changed sources or caption settings get new entries; scoring-only changes can
reuse the same features. Do not modify dataset files while a job is running.
Corrupted complete entries fail with the affected path; move that file aside
to recompute it. Temporary files from interrupted writes are ignored.

Rerun the same command to reuse all completed video checkpoints. The number of
GPUs may change on resume. An interrupted video restarts from its beginning;
there is no within-video checkpoint. One scheduler holds a filesystem lock
on a checkpoint directory. Worker failure stops the other workers and retains
completed checkpoints.

After workers exit, one GPU assembles the normal bank in the original manifest
order using the existing bank builder, then runs the existing scoring and
pooled-metric evaluation from cached videos. This final text-embedding and
scoring phase is **serial**, and is not accelerated by additional workers.
It does not rerun expensive visual processing or window captioning. Optional
`llm_review: true` may load Gemma again during this final phase.

The default bank/output paths are separate from the original evaluator.
An existing compatible bank is reused; it is never overwritten.
`--stage build-memory` only preprocesses normal videos and assembles the bank.
`--stage evaluate` requires an existing bank and only preprocesses test videos.
A new timestamped evaluation output directory is used on each invocation.
If `--output-dir` is supplied, it must be empty: scoring itself has no resume,
but a retry still reuses the costly video checkpoints.

## Resources and validation

The current processor buffers all sampled frames for a video in host RAM.
Several long videos running together can substantially increase host memory
and disk bandwidth demands. `--threads-per-worker` defaults to 4 to limit CPU
thread oversubscription; adjust it to suit the host.

CPU-only tests exercise real spawned workers, failure cleanup, cache
invalidation, atomic publication, scheduling, and serial-versus-cached memory
and metric equivalence with deterministic fixtures:

```bash
python -m pytest tests -q
```

Real-model GPU correctness, hardware utilization, and scaling should be measured
on the multi-GPU machine. Parallel ordering and stochastic model generation can
produce different captions; fixture equivalence is not proof of bit-identical
real-model predictions.
