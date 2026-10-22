"""Own one loopback-only Ollama server per GPU for the lifetime of a run."""

from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)


def resolve_model(config):
    """Pin an already installed model; never pull or modify shared model storage."""
    settings = config.get("reasoning", {})
    if settings.get("backend", "transformers") != "ollama":
        return
    for key, default in (("num_ctx", 4096), ("num_thread", 4), ("timeout", 600), ("startup_timeout", 120)):
        if not isinstance(settings.get(key, default), (int, float)) or settings.get(key, default) <= 0:
            raise ValueError(f"reasoning.{key} must be positive")
    model = settings.get("model", "gemma4:31b-it-q8_0")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)?(:[A-Za-z0-9_.-]+)?", model):
        raise ValueError("Use an installed Ollama model name, e.g. gemma4:31b-it-q8_0")
    name, separator, tag = model.partition(":")
    tag = tag if separator else "latest"
    parts = name.split("/")
    if len(parts) == 1:
        parts.insert(0, "library")
    models = Path(settings.get("models_dir") or os.environ.get("OLLAMA_MODELS")
                  or Path.home() / ".ollama/models").expanduser().resolve()
    manifest_path = models / "manifests/registry.ollama.ai" / parts[0] / parts[1] / tag
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    for layer in [manifest["config"], *manifest["layers"]]:
        digest = layer["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Invalid Ollama layer digest")
        blob = models / "blobs" / digest.replace(":", "-")
        if not blob.is_file() or blob.stat().st_size != layer["size"]:
            raise ValueError(f"Missing/incomplete Ollama model blob: {blob}")
    details = json.loads((models / "blobs" / manifest["config"]["digest"].replace(":", "-")).read_text())
    settings.update(model=f"{name}:{tag}", models_dir=str(models),
                    model_digest=hashlib.sha256(raw).hexdigest(),
                    quantization=details.get("file_type", "unknown"))
    config["reasoning"] = settings


def caption_spec(config):
    """Stable identity excludes endpoint ports, paths to logs, and GPU count."""
    settings = config.get("reasoning", {})
    if settings.get("backend", "transformers") != "ollama":
        return None
    if not settings.get("model_digest"):
        raise ValueError("Resolve the installed Ollama model before using checkpoints")
    return {"backend": "ollama", "model": settings["model"],
            "digest": settings["model_digest"], "quantization": settings["quantization"],
            "generation": "ollama_chat_png_v1", "think": False,
            "temperature": settings.get("temperature", 0), "seed": settings.get("seed", 0),
            "num_ctx": settings.get("num_ctx", 4096)}


def _stop(process):
    # The runner inherits this server's process group; never signal other Ollama servers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@contextmanager
def ollama_servers(devices, config, log_dir):
    settings = config.get("reasoning", {})
    if settings.get("backend", "transformers") != "ollama":
        yield
        return
    executable = shutil.which(settings.get("executable", "ollama"))
    if not executable:
        raise RuntimeError("ollama executable not found on PATH")
    base_port = int(settings.get("base_port", 11500))
    if base_port < 1024 or base_port + len(devices) > 65536:
        raise ValueError("Invalid Ollama base_port")
    # Refuse occupied ports rather than sending requests to somebody else's server.
    for index in range(len(devices)):
        with socket.socket() as probe:
            # A finished smoke test can leave TCP TIME_WAIT entries on this port.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", base_port + index))
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    processes, logs, hosts = [], [], {}
    previous = config.get("_ollama_hosts")
    try:
        for index, device in enumerate(devices):
            host = f"http://127.0.0.1:{base_port + index}"
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=device, OLLAMA_HOST=host,
                       OLLAMA_MODELS=settings["models_dir"], OLLAMA_NUM_PARALLEL="1",
                       OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="-1",
                       OLLAMA_LOAD_TIMEOUT=settings.get("load_timeout", "15m"),
                       OLLAMA_NO_CLOUD="1", OLLAMA_VULKAN="0")
            log_path = folder / f"gpu_{index}.log"
            handle = log_path.open("a")
            logs.append(handle)
            process = subprocess.Popen([executable, "serve"], env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            deadline = time.monotonic() + settings.get("startup_timeout", 120)
            # Ignore proxy environment variables for these strictly local requests.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Ollama on GPU {device} exited; inspect {log_path}")
                try:
                    with opener.open(host + "/api/version", timeout=1) as response:
                        json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError, OSError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Ollama startup timed out; inspect {log_path}")
                    time.sleep(.2)
            hosts[device] = host
            LOGGER.info("Ollama GPU %s: %s (PID %d; log %s)", device, host, process.pid, log_path)
        config["_ollama_hosts"] = hosts
        yield
    finally:
        for process in reversed(processes):
            _stop(process)
        for handle in logs:
            handle.close()
        if previous is None:
            config.pop("_ollama_hosts", None)
        else:
            config["_ollama_hosts"] = previous
