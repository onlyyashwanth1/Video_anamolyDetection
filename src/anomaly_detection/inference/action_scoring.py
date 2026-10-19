"""Frozen CLIP action evidence. Scores are heuristic, not probabilities."""

import math
import numpy as np


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
        if not normal or not anomalous:
            raise ValueError("Both normal and anomalous prompts are required")
        self.normal = np.stack([encoder.encode_text(t) for t in normal])
        self.anomalous = np.stack([encoder.encode_text(t) for t in anomalous])
        self.temperature = float(self.settings.get("temperature", 0.1))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Action temperature must be finite and positive")

    def score(self, event):
        def margin(vector):
            return float(np.max(self.anomalous @ vector) - np.max(self.normal @ vector))
        image_margin = margin(event.average_embedding)
        text_margin = margin(self.encoder.encode_text(event.description)) if event.description else image_margin
        # Transform the two probes separately before combining. Tune on validation data.
        def bounded(value):
            return 1 / (1 + math.exp(-max(-60, min(60, value / self.temperature))))
        return {"action_image_margin": image_margin, "action_text_margin": text_margin,
                "action_score": (bounded(image_margin) + bounded(text_margin)) / 2}
