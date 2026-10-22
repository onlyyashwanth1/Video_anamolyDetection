# UCF-Crime with quantized Gemma in a detached screen

This setup uses the locally installed `gemma4:31b-it-q8_0` model, the existing
CLIP/YOLO pipeline, and the dataset under
`/home/atripathi/prudhvi/Dataset/UCFCrime`. No full-precision Gemma download or
Transformers upgrade is needed. Settings are in `config/ucf_ollama.yaml`.

## Start

From the repository directory, choose GPUs available to your job:

```bash
bash scripts/run_ucf_screen.sh --gpus 0,1,2,3,7 --stage all
```

The launcher first validates model files, annotations and video metadata, then
starts a detached GNU screen named `ucf-ollama`. You can close the SSH/remote
connection after the launcher prints its session and log path. Screen keeps the
job running while the server stays up. It does not survive a server reboot.

```bash
screen -ls
screen -r ucf-ollama
```

Detach without stopping the job: press **Ctrl+A**, release, then press **D**.
Logs are written continuously to
`outputs/screen_runs/ucf-ollama-<timestamp>-<pid>/evaluation.log`.
Monitor without attaching:

```bash
tail -f outputs/screen_runs/ucf-ollama-*/evaluation.log
nvidia-smi
```

Each run folder records the evaluator PID and an `exit_code` when finished
(`0` means success). The screen session ends when the evaluator exits; logs and
results remain. If the session disappears, check its log and exit code.

## GPU routing and memory

Every selected GPU has one video worker (CLIP + YOLO) and one dedicated Ollama
server with its own quantized Gemma replica. Servers listen only on loopback,
starting at port 11500. GPU selection is set on the server process itself;
setting it only on a Python client would not control Ollama's GPUs.
When `CUDA_VISIBLE_DEVICES` is inherited, `--gpus` indexes that allocation,
as in the existing multi-GPU evaluator.

Startup checks require vision support, the expected installed model digest and
full GPU residency. No CPU offload or silent model substitution is allowed.
Model thinking is disabled, temperature/seed are zero, and context length is
4096 tokens. Ollama CPU threads default to four per replica (`reasoning.num_thread`).
Captions use lossless PNG image bytes. A fresh memory bank is
needed when switching from full-precision Gemma to the quantized model;
quantization can change captions and resulting metrics.

The evaluator owns its Ollama servers and stops them on normal exit, worker
failure, Ctrl+C, SIGTERM or SIGHUP. Existing Ollama services are not reused or
stopped. It refuses occupied ports; change `reasoning.base_port` to use a
different block. Avoid changing the installed model tag while a run is active.
Per-server logs are under `outputs/ollama_logs/ucf/<timestamp>/`.

The pipeline buffers all sampled frames for each video in host RAM, as before.
The longest normal video exceeds nine hours; its first checkpoint may take a
long time. A heartbeat reports active videos every minute. Additional GPUs
accelerate video processing; final memory assembly and scoring use one GPU.

## Stop and resume

Attach with `screen -r ucf-ollama`, then press **Ctrl+C** to stop.
Alternatively send SIGTERM to the PID in the current run's `evaluator.pid`.
Completed video checkpoints remain reusable; an interrupted video starts again
from its beginning. Once the old session exits, rerun the same launcher command
to resume. The GPU count may change. Resume uses:

- Checkpoints: `outputs/checkpoints/ucf_ollama/`
- Normal memory: `outputs/memory/ucf_ollama.npz`
- Results: `outputs/ucf_multi_gpu_evaluation/<timestamp>/`

The checkpoint identity includes model tag, manifest digest, quantization,
generation settings, and the existing video/feature settings. Changing GPU IDs,
server ports or thread counts does not invalidate completed video features.

## Validate or run without screen

```bash
# No model/GPU loading; validates the installed local model and dataset.
.venv/bin/python scripts/evaluate_ucf_multi_gpu.py \
  --config config/ucf_ollama.yaml --gpus 0,1 --dry-run

# Check a real frame and a text summary using one GPU.
.venv/bin/python scripts/smoke_ucf_ollama.py --gpus 0 --models-only

# Process two complete short videos on separate GPUs and save reusable checkpoints.
# Also build a separate tiny smoke-test memory bank and verify final metrics/plots.
.venv/bin/python scripts/smoke_ucf_ollama.py --gpus 0,1

# Foreground evaluation, using the same artifacts as the screen launcher.
.venv/bin/python -u scripts/evaluate_ucf_multi_gpu.py \
  --config config/ucf_ollama.yaml --gpus 0,1 --stage all \
  --checkpoint-dir outputs/checkpoints/ucf_ollama \
  --memory-bank-path outputs/memory/ucf_ollama.npz
```

Ollama must be on PATH and the model must already be installed in
`~/.ollama/models` (or `OLLAMA_MODELS` / `reasoning.models_dir`). The launcher
never pulls models. Optional `VAD_PYTHON` selects another Python executable and
`VAD_SESSION` changes the screen name. Install project dependencies with
`python -m pip install -r requirements.txt`; pytest is only needed for tests.
The original Transformers configuration and commands remain available.
