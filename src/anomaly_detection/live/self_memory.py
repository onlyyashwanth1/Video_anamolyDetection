"""Normal memory built from the start of the same video.

Each window has two embeddings: its mean CLIP image feature and a sentence
embedding of its Gemma caption. Novelty is the distance to the nearest
memory windows (0.6 * top-k mean + 0.4 * best match, as in visual_memory.py).

Distances are first turned into "how many normal spreads above normal" (robust z)
using the observation period itself. Each observation window is compared
with the memory after removing the windows close to it in time, so it
cannot match itself or its overlapping neighbours. This is a held-out
estimate of how far ordinary, not-yet-seen moments of this scene sit
from memory.

Every component is then a 0-1 score (0 = normal, 1 = anomalous), as in the
main project: visual and caption novelty use sigmoid(z - 3), so 3 normal
spreads above normal gives 0.5; the Gemma judge rating 0-9 becomes rating / 9.
The anomaly score is their weighted mean. With equal weights one component
alone reaches at most 1/3, so a score above 0.5 needs at least two to agree.
"""

from dataclasses import dataclass, field

import numpy as np

SIGMA_PER_IQR = 1.349  # IQR of a normal distribution in standard deviations
Z_AT_HALF = 3.0        # a component scores 0.5 at 3 normal spreads above normal
JUDGE_MAX = 9.0        # judge ratings are 0-9


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(np.asarray(values, dtype=float), -60, 60)))


def boundary_frame(frame_count, fps, fraction=None, seconds=None):
    """Native frame where observation ends and testing starts."""
    if (fraction is None) == (seconds is None):
        raise ValueError("Give exactly one of observe fraction or observe seconds")
    if fraction is not None:
        if not 0 < fraction < 1:
            raise ValueError("observe fraction must be between 0 and 1")
        return int(frame_count * fraction)
    if seconds <= 0:
        raise ValueError("observe seconds must be positive")
    return min(int(round(seconds * fps)), frame_count - 1)


def split_windows(events, boundary):
    """Memory windows end by the boundary; test windows start at or after it.

    The few windows that cross the boundary are used for neither, so no
    test frame ever enters memory.
    """
    memory = [e for e in events if e.stop_frame <= boundary]
    test = [e for e in events if e.start_frame >= boundary]
    if not memory or not test:
        raise ValueError("Video too short for separate observation and test windows")
    return memory, test


def unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(norms <= 1e-12):
        raise ValueError("Embeddings must be finite and nonzero")
    return matrix / norms


def knn_distance(queries, memory, k, excluded=None):
    """Return (distance, index of the nearest memory row) for each query row.

    ``excluded`` is a (queries x memory) boolean mask of rows that may not be used.
    """
    sims = np.asarray(queries) @ np.asarray(memory).T
    if excluded is not None:
        sims = np.where(excluded, -np.inf, sims)
    distances, nearest = np.empty(len(sims)), np.empty(len(sims), dtype=int)
    for row, values in enumerate(sims):
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise ValueError("No memory windows left after the time guard; shorten the guard")
        top = np.sort(valid)[::-1][:min(k, len(valid))]
        distances[row] = 1 - (.6 * top.mean() + .4 * top[0])
        nearest[row] = int(np.argmax(values))
    return distances, nearest


def time_guard(events, guard_frames):
    """True where two windows overlap or lie within ``guard_frames`` of each other."""
    starts = np.asarray([e.start_frame for e in events])
    stops = np.asarray([e.stop_frame for e in events])
    return ((starts[None, :] < stops[:, None] + guard_frames)
            & (starts[:, None] < stops[None, :] + guard_frames))


@dataclass
class RobustScale:
    """Median and IQR-based spread of the held-out observation values."""
    center: float
    scale: float

    @classmethod
    def fit(cls, values, floor):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Calibration needs finite values")
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        return cls(float(median), float(max((q75 - q25) / SIGMA_PER_IQR, floor)))

    def z(self, values):
        return (np.asarray(values, dtype=float) - self.center) / self.scale


@dataclass
class SelfMemory:
    """Memory of one scene's normal windows, frozen after observation."""
    visual: np.ndarray
    text: np.ndarray
    captions: list
    weights: dict
    k: int
    centering: bool
    visual_mean: np.ndarray = None
    text_mean: np.ndarray = None
    scales: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    @classmethod
    def build(cls, memory_events, visual, text, settings, fps, judge_scores=None):
        """Store memory windows and calibrate on held-out observation windows."""
        k = int(settings.get("k", 10))
        weights = {"visual": float(settings.get("visual_weight", 1.0)),
                   "text": float(settings.get("text_weight", 1.0)),
                   "judge": float(settings.get("judge_weight", 0.0)) if judge_scores is not None else 0.0}
        if k < 1 or any(not np.isfinite(w) or w < 0 for w in weights.values()) or not sum(weights.values()):
            raise ValueError("k must be positive and weights nonnegative with a positive sum")
        if len(memory_events) < int(settings.get("min_memory_windows", 8)):
            raise ValueError(f"Only {len(memory_events)} observation windows; use a longer video "
                             "or a larger observation fraction")
        centering = bool(settings.get("centering", False))
        visual, text = unit_rows(visual), unit_rows(text)
        model = cls(visual=visual, text=text, captions=[e.description for e in memory_events],
                    weights=weights, k=k, centering=centering)
        if centering:
            # Remove what every moment of this scene shares, keep what changes.
            model.visual_mean, model.text_mean = visual.mean(0), text.mean(0)
            model.visual = unit_rows(visual - model.visual_mean)
            model.text = unit_rows(text - model.text_mean)
        guard = time_guard(memory_events, int(round(float(settings.get("calibration_guard_seconds", 4.0)) * fps)))
        held_out = {
            "visual": knn_distance(model.visual, model.visual, k, guard)[0],
            "text": knn_distance(model.text, model.text, k, guard)[0],
        }
        model.scales = {name: RobustScale.fit(values, 1e-3) for name, values in held_out.items()}
        scores = {name: sigmoid(model.scales[name].z(values) - Z_AT_HALF) for name, values in held_out.items()}
        if weights["judge"]:
            scores["judge"] = model._judge_score(judge_scores)
        model.calibration = {
            "held_out_windows": len(memory_events),
            "guard_seconds": float(settings.get("calibration_guard_seconds", 4.0)),
            **{f"{name}_median": s.center for name, s in model.scales.items()},
            **{f"{name}_spread": s.scale for name, s in model.scales.items()},
            "observation_max_score": float(model._fuse(scores).max()),
        }
        return model

    @staticmethod
    def _judge_score(ratings):
        ratings = np.asarray(ratings, dtype=float)
        if not np.isfinite(ratings).all():
            raise ValueError("Judge ratings must be finite")
        return np.clip(ratings / JUDGE_MAX, 0, 1)

    def _fuse(self, scores):
        """Weighted mean of 0-1 component scores."""
        total = sum(self.weights[name] for name in scores)
        return sum(self.weights[name] * scores[name] for name in scores) / total

    def _prepare(self, vectors, mean):
        vectors = unit_rows(np.atleast_2d(vectors))
        return unit_rows(vectors - mean) if self.centering else vectors

    def score(self, visual, text, judge=None):
        """Score test windows; returns per-window arrays and the nearest memory caption."""
        visual_d, nearest = knn_distance(self._prepare(visual, self.visual_mean), self.visual, self.k)
        text_d, _ = knn_distance(self._prepare(text, self.text_mean), self.text, self.k)
        zs = {"visual": self.scales["visual"].z(visual_d), "text": self.scales["text"].z(text_d)}
        scores = {name: sigmoid(z - Z_AT_HALF) for name, z in zs.items()}
        if self.weights["judge"]:
            if judge is None:
                raise ValueError("This memory was calibrated with judge ratings; pass them")
            scores["judge"] = self._judge_score(judge)
        return {"visual_distance": visual_d, "text_distance": text_d,
                **{f"{name}_z": value for name, value in zs.items()},
                **{f"{name}_score": value for name, value in scores.items()},
                "score": self._fuse(scores),
                "nearest_memory_caption": [self.captions[i] for i in nearest]}
