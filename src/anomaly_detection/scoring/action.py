"""Action architecture: CLIP prompt evidence, optionally mixed with saved normal memory.

ActionScorer compares a window's CLIP image embedding and its caption's CLIP
text embedding with lists of normal and anomalous action prompts. Scores are
heuristic, not probabilities.

ActionMemoryScorer is the architecture:
    score = (1 - memory_weight) * action + memory_weight * calibrated memory
memory_weight 0.2 (80/20) is the report's action + memory baseline; 0 is pure
action scoring and then no memory bank is loaded. The shared_best configuration is this architecture with
memory_weight 0 and its own prompt settings.
"""

import math
import numpy as np

from anomaly_detection.datatypes import AnomalyResult


DEFAULT_NORMAL = [
    "people talking peacefully", "people walking normally", "people playing sports normally",
    "vehicles moving normally on a road", "people carrying objects", "a quiet scene without dangerous activity",
]
DEFAULT_ANOMALOUS = [
    "a person punching or kicking another person", "a person firing a gun",
    "vehicles colliding in a road accident", "a fire or explosion",
    "a person smashing property", "a person forcibly taking another person's belongings",
]


class ActionScorer:
    def __init__(self, encoder, settings=None):
        self.encoder = encoder
        self.settings = settings or {}
        normal = self.settings.get("normal_prompts", DEFAULT_NORMAL)
        anomalous = self.settings.get("anomalous_prompts", DEFAULT_ANOMALOUS)
        self.temperature = float(self.settings.get("temperature", 0.1))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Action temperature must be finite and positive")
        self.image_temperature = self._positive_setting("image_temperature", self.temperature)
        self.text_temperature = self._positive_setting("text_temperature", self.temperature)
        self.prompt_reduction_temperature = self._positive_setting("prompt_reduction_temperature", .05)
        self.image_weight = float(self.settings.get("image_weight", 0.5))
        if not math.isfinite(self.image_weight) or not 0 <= self.image_weight <= 1:
            raise ValueError("Action image_weight must be finite and in [0, 1]")
        self.action_transform = self.settings.get("action_transform", "before_fusion")
        if self.action_transform not in ("before_fusion", "after_fusion"):
            raise ValueError("Action action_transform must be before_fusion or after_fusion")
        self.image_prompt_reducer = self.settings.get("image_prompt_reducer", "max")
        self.text_prompt_reducer = self.settings.get("text_prompt_reducer", "max")
        reducers = ("max", "mean", "top2", "logmeanexp")
        if self.image_prompt_reducer not in reducers or self.text_prompt_reducer not in reducers:
            raise ValueError("Action prompt reducers must be max, mean, top2 or logmeanexp")

        # Different probes may need different prompt wording, while the exact
        # same probe definitions remain usable on every evaluation dataset.
        groups = {}
        for probe in ("image", "text"):
            for kind, fallback in (("normal", normal), ("anomalous", anomalous)):
                key = f"{probe}_{kind}_prompts"
                group = self.settings.get(key, fallback)
                if (not isinstance(group, (list, tuple)) or not group
                        or any(not isinstance(t, str) or not t.strip() for t in group)
                        or len(set(group)) != len(group)):
                    raise ValueError("Action prompts must be lists of distinct nonempty strings")
                groups[key] = tuple(group)
                setattr(self, key, tuple(group))
        encoded = {text: np.asarray(encoder.encode_text(text))
                   for text in dict.fromkeys(text for group in groups.values() for text in group)}
        dimensions = {v.shape for v in encoded.values()}
        if (len(dimensions) != 1 or any(v.ndim != 1 or not v.size or not np.isfinite(v).all()
                                        for v in encoded.values())):
            raise ValueError("Action prompt embeddings must be aligned finite vectors")
        for key, group in groups.items():
            setattr(self, key.removesuffix("_prompts"), np.stack([encoded[text] for text in group]))
        # Public legacy aliases retained for callers inspecting the image probe.
        self.normal, self.anomalous = self.image_normal, self.image_anomalous

    def _positive_setting(self, name, default):
        value = float(self.settings.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Action {name} must be finite and positive")
        return value

    def _reduce(self, similarities, method):
        if method == "max":
            return np.max(similarities, axis=1)
        if method == "mean":
            return np.mean(similarities, axis=1)
        if method == "top2":
            k = min(2, similarities.shape[1])
            return np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
        # Log of the mean, rather than the sum, keeps the reduction comparable
        # when the normal and anomalous prompt lists have different lengths.
        peak = np.max(similarities, axis=1, keepdims=True)
        reduced = np.log(np.exp((similarities - peak) / self.prompt_reduction_temperature).mean(axis=1))
        return peak[:, 0] + self.prompt_reduction_temperature * reduced

    def _margin(self, vectors, normal, anomalous, reducer):
        return (self._reduce(vectors @ anomalous.T, reducer)
                - self._reduce(vectors @ normal.T, reducer)).astype(float)

    @staticmethod
    def _bounded(logits):
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))

    def score(self, event):
        images = np.asarray(event.average_embedding)[None, :]
        texts = np.asarray(self.encoder.encode_text(event.description))[None, :] if event.description else None
        return {key: float(value[0]) for key, value in self.score_vectors(images, texts).items()}

    def score_vectors(self, images, texts=None):
        """Score cached window embeddings without invoking an encoder per window."""
        images = np.asarray(images)
        texts = None if texts is None else np.asarray(texts)
        if (images.ndim != 2 or not len(images)
                or images.shape[1] != self.normal.shape[1]
                or not np.isfinite(images).all()
                or (texts is not None and (texts.shape != images.shape or not np.isfinite(texts).all()))):
            raise ValueError("Expected aligned finite image/text embedding rows")
        image_margin = self._margin(images, self.image_normal, self.image_anomalous, self.image_prompt_reducer)
        image_logits = image_margin / self.image_temperature
        image_score = self._bounded(image_logits)
        if texts is None:
            # No caption: use only the image evidence, including its scale.
            text_margin, text_logits, text_score = image_margin.copy(), image_logits.copy(), image_score.copy()
        else:
            text_margin = self._margin(texts, self.text_normal, self.text_anomalous, self.text_prompt_reducer)
            text_logits = text_margin / self.text_temperature
            text_score = self._bounded(text_logits)
        if self.action_transform == "before_fusion":
            action_score = self.image_weight * image_score + (1 - self.image_weight) * text_score
        else:
            action_score = self._bounded(self.image_weight * image_logits + (1-self.image_weight) * text_logits)
        return {"action_image_margin": image_margin, "action_text_margin": text_margin,
                "action_image_score": image_score, "action_text_score": text_score,
                "action_score": action_score}


class ActionMemoryScorer:
    """Per-window action scores mixed with the saved bank's memory score."""

    def __init__(self, settings, encoder, bank=None):
        self.settings = settings
        self.encoder = encoder
        self.memory_weight = float(settings.get("memory_weight", 0.05))
        if not math.isfinite(self.memory_weight) or not 0 <= self.memory_weight <= 1:
            raise ValueError("action.memory_weight must be in [0, 1]")
        self.memory_image_weight = float(settings.get("memory_image_weight", 1.0))
        self.threshold = float(settings["threshold"])
        if self.memory_weight > 0 and bank is None:
            raise ValueError("action.memory_weight > 0 needs a saved normal-memory bank")
        self.bank = bank if self.memory_weight > 0 else None
        self.action = ActionScorer(encoder, settings)

    @staticmethod
    def needs_bank(settings):
        return float(settings.get("memory_weight", 0.05)) > 0

    def score_event(self, event):
        if event.average_embedding is None:
            raise ValueError("Window has no image embedding")
        components, closest = {}, None
        if self.bank is not None:
            if not event.description:
                raise ValueError("Memory scoring needs a caption")
            components = self.bank.score_memory(event.average_embedding,
                                                self.encoder.encode_text(event.description),
                                                image_weight=self.memory_image_weight, k=self.bank.top_k)
            closest = self.bank.closest_visual(event.average_embedding).text
        components.update(self.action.score(event))
        score = components["action_score"]
        if self.bank is not None:
            score = self.memory_weight * components["memory_score"] + (1 - self.memory_weight) * score
        components["combined_score"] = score
        return AnomalyResult(event=event, anomaly_score=score, is_anomaly=score > self.threshold,
                             closest_match=closest, component_scores=components)

    def score_video(self, events, frame_count=None, fps=None):
        return [self.score_event(event) for event in events]

    @staticmethod
    def unit(scores):
        """Action and memory scores are already in [0, 1]; clip after smoothing."""
        return np.clip(scores, 0, 1)
