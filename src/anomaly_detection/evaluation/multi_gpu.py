"""Single-host GPU workers. Keep module imports free of CUDA/model imports."""

from contextlib import contextmanager
import logging
import math
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback

LOGGER = logging.getLogger(__name__)


def resolve_devices(selection, visible=None):
    """Numeric IDs index the existing CUDA_VISIBLE_DEVICES allocation, if set."""
    requested = [part.strip() for part in selection.split(",")]
    if not requested or any(not part.isdigit() for part in requested):
        raise ValueError("--gpus must be comma-separated nonnegative indices, e.g. 0,1,2")
    indices = [int(part) for part in requested]
    if len(indices) != len(set(indices)):
        raise ValueError("Each GPU must be selected only once")
    if visible is None:
        return [str(i) for i in indices]
    allocation = [part.strip() for part in visible.split(",") if part.strip()]
    if not allocation or "-1" in allocation or max(indices) >= len(allocation):
        raise ValueError("--gpus exceeds the inherited CUDA_VISIBLE_DEVICES allocation")
    devices = [allocation[i] for i in indices]
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate selected devices")
    return devices


@contextmanager
def cache_lock(folder):
    """Linux advisory lock released automatically on exit/crash; no stale PID files."""
    import fcntl

    Path(folder).mkdir(parents=True, exist_ok=True)
    with (Path(folder) / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator is using checkpoint directory {folder}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pending_jobs(videos, cache):
    """Validate checkpoints and probe container metadata only (no frame decoding)."""
    import cv2

    jobs, resumed = [], 0
    for name, source in videos.items():
        if cache.load(source) is not None:
            resumed += 1
            continue
        cap = cv2.VideoCapture(str(source))
        try:
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if not math.isfinite(fps) or not math.isfinite(frames) or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot determine video duration: {source}")
        jobs.append({"name": name, "source": str(source), "seconds": frames / fps})
    return sorted(jobs, key=lambda j: (-j["seconds"], j["name"])), resumed


def _configure_gpu(device, threads):
    # Called in a fresh spawned interpreter BEFORE importing torch or processors.
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    import torch

    torch.set_num_threads(threads)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"GPU {device} is unavailable; refusing CPU fallback")
    torch.cuda.set_device(0)
    LOGGER.info("PID %d: GPU %s -> cuda:0 (%s)", os.getpid(), device, torch.cuda.get_device_name(0))


def _worker(device, config, cache_dir, threads, jobs, messages):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(levelname)s %(message)s")
    current = None
    try:
        _configure_gpu(device, threads)
        import cv2
        from anomaly_detection.processing import VideoEventProcessor
        from anomaly_detection.reasoning.local_gemma import _load
        from anomaly_detection.evaluation.video_cache import VideoCache

        cv2.setNumThreads(threads)
        cache = VideoCache(cache_dir, config)
        processor = VideoEventProcessor(config, strict=True)
        processor.masker.model.to("cuda:0")
        _, gemma = _load()
        if any(p.device.type != "cuda" for p in gemma.parameters()):
            raise RuntimeError("Gemma did not fit on one GPU; CPU/disk offload is not supported here")
        while True:
            current = jobs.get()
            if current is None:
                break
            messages.put(("started", device, current["name"]))
            started = time.monotonic()
            identity = cache.identity(current["source"])
            video = processor.process(current["source"])
            cache.save(current["source"], video, expected_identity=identity)
            del video
            messages.put(("done", device, current["name"], time.monotonic() - started))
    except BaseException:
        messages.put(("error", device, current, traceback.format_exc()))
        raise


def run_workers(devices, config, cache_dir, jobs, threads=4, worker_target=None):
    """A shared FIFO queue gives the next longest video to the next free GPU."""
    if not jobs:
        return
    context = multiprocessing.get_context("spawn")
    tasks, messages = context.Queue(), context.Queue()
    workers = []
    done, active = set(), {}
    try:
        for job in jobs:
            tasks.put(job)
        for device in devices[:len(jobs)]:
            tasks.put(None)
            process = context.Process(target=worker_target or _worker,
                                      args=(device, config, str(cache_dir), threads, tasks, messages))
            process.start()
            workers.append(process)
        last_report = time.monotonic()
        while len(done) < len(jobs):
            try:
                message = messages.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in workers):
                    raise RuntimeError("A GPU worker exited unexpectedly; completed checkpoints are retained")
                if all(not p.is_alive() for p in workers):
                    raise RuntimeError("Workers exited without completing every video")
                if time.monotonic() - last_report >= 60:
                    LOGGER.info("Completed %d/%d; active videos: %s", len(done), len(jobs), active)
                    last_report = time.monotonic()
                continue
            kind, device = message[:2]
            if kind == "error":
                raise RuntimeError(f"GPU {device} failed: {message[2]}\n{message[3]}")
            if kind == "started":
                active[device] = message[2]
                LOGGER.info("GPU %s starting %s", device, message[2])
            elif kind == "done":
                done.add(message[2])
                active.pop(device, None)
                LOGGER.info("Completed %d/%d: %s on GPU %s in %.1fs",
                            len(done), len(jobs), message[2], device, message[3])
        for process in workers:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"Worker {process.pid} did not exit successfully")
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        # Do not block flushing queued jobs after a failed/terminated worker.
        for channel in (tasks, messages):
            channel.cancel_join_thread()
            channel.close()


def _finalize(device, config, cache_dir, normal, testing, annotations,
              bank_path, annotation_path, output, stage, threads):
    """Only CLIP is needed here unless optional test-time Gemma review is enabled."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _configure_gpu(device, threads)
    from anomaly_detection.encoding.multi_scale_encoder import MultiScaleClipEncoder
    from anomaly_detection.evaluation.video_cache import VideoCache, CachedProcessor
    from anomaly_detection.evaluation.bank import build_bank, load_bank

    processor = CachedProcessor(VideoCache(cache_dir, config),
                                MultiScaleClipEncoder(config["encoder"]["model_name"], device="cuda:0"))
    if Path(bank_path).exists():
        bank, metadata = load_bank(bank_path, config, normal)
    else:
        bank, metadata = build_bank(processor, normal, config, bank_path)
    if stage != "build-memory":
        from anomaly_detection.evaluation.runner import evaluate
        evaluate(processor, bank, metadata, testing, annotations, config,
                 bank_path, annotation_path, output)


def finalize(*args):
    # Parent never initializes CUDA. Spawn also releases worker models before merging.
    process = multiprocessing.get_context("spawn").Process(target=_finalize, args=args)
    process.start()
    try:
        while process.is_alive():
            process.join(timeout=1)
        if process.exitcode:
            raise RuntimeError("Memory assembly/evaluation failed; video checkpoints remain reusable")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
