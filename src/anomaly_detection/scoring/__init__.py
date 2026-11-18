"""The three architectures, selected by the config's `architecture:` key.

observe_detect  basic observe-then-detect: memory from the first part of the same
                video (single videos only; see observe_detect.run_observe_detect).
action          CLIP action prompts mixed with the saved normal-memory bank
                (action.memory_weight; 0 = prompts only, no bank).
memory_novelty  B: novelty against the saved normal-memory bank;
                memory_novelty.judge_weight > 0 adds the memory-aware Gemma judge (B+J).

Window scorers (action, memory_novelty) share one interface:
    score_video(events, frame_count, fps) -> list[AnomalyResult]  (raw score per window)
    unit(scores) -> scores in [0, 1], applied after window -> frame projection
    threshold    alert cutoff on the unit scale
"""

from pathlib import Path
import re

ARCHITECTURES = ("observe_detect", "action", "memory_novelty")
ROOT = Path(__file__).resolve().parents[3]


def architecture(config):
    name = config.get("architecture")
    if name not in ARCHITECTURES:
        raise ValueError(f"config 'architecture' must be one of {ARCHITECTURES}, not {name!r}")
    return name


def needs_bank(config):
    name = architecture(config)
    if name == "action":
        from anomaly_detection.scoring.action import ActionMemoryScorer
        return ActionMemoryScorer.needs_bank(config["action"])
    return name == "memory_novelty"


def uses_judge(config):
    return (architecture(config) == "memory_novelty"
            and float(config["memory_novelty"].get("judge_weight", 0)) > 0)


def judge_cache_path(config):
    settings = config["memory_novelty"].get("judge", {})
    if settings.get("cache_path"):
        path = Path(settings["cache_path"]).expanduser()
        return path if path.is_absolute() else ROOT / path
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", config["reasoning"]["model"])
    return ROOT / "outputs/judge_cache" / f"{model}_ctx{int(settings.get('num_ctx', 2048))}.json"


def build_scorer(config, encoder, bank=None, bank_path=None, text_encoder=None):
    """Window scorer for the configured architecture (not observe_detect)."""
    name = architecture(config)
    if name == "action":
        from anomaly_detection.scoring.action import ActionMemoryScorer
        return ActionMemoryScorer(config["action"], encoder, bank)
    if name == "memory_novelty":
        from anomaly_detection.scoring.judge import MemoryJudge, OllamaRater, RatingCache
        from anomaly_detection.scoring.memory_novelty import MemoryNoveltyScorer, NoveltyMemory
        if bank is None:
            raise ValueError("The memory_novelty architecture needs a saved normal-memory bank")
        settings = config["memory_novelty"]
        device = config.get("memory_bank", {}).get("search_device", "cpu")
        if text_encoder is None:
            from anomaly_detection.perception.text_encoder import DEFAULT_MODEL, SentenceEncoder
            text_encoder = SentenceEncoder(settings.get("text_encoder", DEFAULT_MODEL),
                                           device=None if device == "cpu" else device)
        alpha = float(settings.get("centering_alpha", 1.0))
        cache = None
        if bank_path is not None:
            encoder_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", getattr(text_encoder, "name", "text"))
            cache = Path(bank_path).with_name(f"{Path(bank_path).stem}.novelty_{encoder_tag}_a{alpha:g}.json")
        memory = NoveltyMemory(bank, text_encoder, alpha=alpha, device=device, cache_path=cache)
        judge = None
        if uses_judge(config):
            judge_settings = settings.get("judge", {})
            rater = (OllamaRater(config, judge_settings.get("num_ctx", 2048))
                     if config.get("_ollama_hosts") else None)
            judge = MemoryJudge(judge_settings, memory, text_encoder, rater,
                                RatingCache(judge_cache_path(config)))
        return MemoryNoveltyScorer(settings, memory, text_encoder, judge)
    raise ValueError("observe_detect builds its memory per video; use scoring.observe_detect.run_observe_detect")
