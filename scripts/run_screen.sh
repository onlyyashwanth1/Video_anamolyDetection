#!/usr/bin/env bash
# Run scripts/evaluate.py detached in GNU screen, logging to outputs/screen_runs/.
# Usage: bash scripts/run_screen.sh --dataset ucf --architecture action_default --gpus 0,1,2,3 [options]
set -euo pipefail
VAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VAD_PYTHON="${VAD_PYTHON:-$VAD_ROOT/.venv/bin/python}"
VAD_SESSION="${VAD_SESSION:-vad-eval}"
cd "$VAD_ROOT"

if [[ "${1:-}" == "--inside-screen" ]]; then
    shift
    VAD_RUN_DIR="$1"
    shift
    exec > >(tee -a "$VAD_RUN_DIR/evaluation.log") 2>&1
    echo "Started: $(date --iso-8601=seconds)"
    printf 'Command:'
    printf ' %q' "$VAD_PYTHON" -u scripts/evaluate.py "$@"
    printf '\n'
    "$VAD_PYTHON" -u scripts/evaluate.py "$@" &
    VAD_JOB_PID=$!
    echo "$VAD_JOB_PID" > "$VAD_RUN_DIR/evaluator.pid"
    trap 'kill -TERM "$VAD_JOB_PID" 2>/dev/null || true' HUP INT TERM
    set +e
    wait "$VAD_JOB_PID"
    VAD_STATUS=$?
    # A trapped signal may interrupt wait before the evaluator has cleaned up.
    if kill -0 "$VAD_JOB_PID" 2>/dev/null; then
        wait "$VAD_JOB_PID"
        VAD_STATUS=$?
    fi
    echo "$VAD_STATUS" > "$VAD_RUN_DIR/exit_code"
    echo "Finished: $(date --iso-8601=seconds); exit code: $VAD_STATUS"
    exit "$VAD_STATUS"
fi

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_screen.sh --dataset ucf --architecture action_default --gpus 0,1,2,3 [evaluate.py options]"
    echo "Optional environment: VAD_SESSION=vad-eval VAD_PYTHON=/path/to/python"
    exit 0
fi
command -v screen >/dev/null || { echo 'GNU screen is required.' >&2; exit 1; }
command -v ollama >/dev/null || { echo 'ollama is required on PATH.' >&2; exit 1; }
[[ -x "$VAD_PYTHON" ]] || { echo "Python not found: $VAD_PYTHON" >&2; exit 1; }
[[ "$VAD_SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid screen session name.' >&2; exit 1; }
if screen -S "$VAD_SESSION" -Q windows >/dev/null 2>&1; then
    echo "Screen session $VAD_SESSION already exists; attach with: screen -r $VAD_SESSION" >&2
    exit 1
fi
# Validate local model files, annotations and all video metadata before detaching.
"$VAD_PYTHON" scripts/evaluate.py "$@" --dry-run
VAD_RUN_DIR="$VAD_ROOT/outputs/screen_runs/$VAD_SESSION-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$VAD_RUN_DIR"
export VAD_PYTHON VAD_SESSION
screen -dmS "$VAD_SESSION" bash "$VAD_ROOT/scripts/run_screen.sh" --inside-screen "$VAD_RUN_DIR" "$@"
echo "Detached screen: $VAD_SESSION"
echo "Attach: screen -r $VAD_SESSION"
echo "Log: $VAD_RUN_DIR/evaluation.log"
echo "Exit status will be saved in: $VAD_RUN_DIR/exit_code"
