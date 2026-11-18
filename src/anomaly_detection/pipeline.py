"""Single-video entry point for all three architectures.

run_full_pipeline(source, config) -> (normal captions, memory bank, per-window results)

observe_detect builds its memory from the start of this video and scores the rest.
action and memory_novelty score every window against the saved normal-memory
bank named by evaluation.memory_bank_path (action with memory_weight 0 needs none).
"""

from pathlib import Path
from typing import List, Tuple

from anomaly_detection.config import load_config  # noqa: F401  (re-exported for scripts)
from anomaly_detection.datatypes import AnomalyResult
from anomaly_detection.memory.bank import MemoryBank
from anomaly_detection.perception.processor import VideoEventProcessor
from anomaly_detection.reasoning.explain import explain_anomaly
from anomaly_detection.scoring import architecture, build_scorer, needs_bank
from anomaly_detection.scoring.observe_detect import run_observe_detect

ROOT = Path(__file__).resolve().parents[2]


def bank_path(config):
    path = Path(config["evaluation"]["memory_bank_path"]).expanduser()
    return path if path.is_absolute() else ROOT / path


def run_full_pipeline(video_source, config) -> Tuple[List[str], MemoryBank, List[AnomalyResult]]:
    from anomaly_detection.memory import store
    from anomaly_detection.reasoning import ollama_runtime

    architecture(config)
    bank = None
    if needs_bank(config):
        # Load the saved bank before starting any caption models.
        ollama_runtime.resolve_model(config)
        bank, _ = store.load_bank(bank_path(config), config)
    with ollama_runtime.caption_session(config):
        return _run(video_source, config, bank)


def _run(video_source, config, bank=None):
    processor = VideoEventProcessor(config)
    video = processor.process(video_source, caption=False)
    if architecture(config) == "observe_detect":
        settings = config["observe_detect"]
        review = explain_anomaly if settings.get("llm_review", False) else None
        return run_observe_detect(video, processor, settings,
                                  config.get("temporal", {}).get("mode", "events"), review)
    if not video.events:
        raise ValueError("No video windows to score")
    processor.caption_events(video.events, video.frames)
    scorer = build_scorer(config, processor.encoder, bank, bank_path(config) if bank is not None else None)
    constitution = [entry.text for entry in bank.entries] if bank is not None else []
    review = config.get("evaluation", {}).get("llm_review", False)
    results = []
    for result in scorer.score_video(video.events, video.frame_count, video.fps):
        result.is_anomaly = bool(scorer.unit(result.anomaly_score) > scorer.threshold)
        if result.is_anomaly and review:
            result = explain_anomaly(result, constitution)
        results.append(result)
    return constitution, bank if bank is not None else MemoryBank(), results
