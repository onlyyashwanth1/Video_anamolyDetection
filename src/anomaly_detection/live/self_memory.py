"""Normal memory built from the start of the same video.

Each window has two embeddings: its mean CLIP image feature and a sentence
embedding of its Gemma caption. Novelty is the distance to the nearest
memory windows (0.6 * top-k mean + 0.4 * best match, as in visual_memory.py).

Distances are turned into "how many normal spreads above normal" (robust z)
using the observation period itself. Each observation window is compared
with the memory after removing the windows close to it in time, so it
cannot match itself or its overlapping neighbours. This is a held-out
estimate of how far ordinary, not-yet-seen moments of this scene sit
from memory.
"""

from dataclasses import dataclass, field

import numpy as np

SIGMA_PER_IQR = 1.349  # IQR of a normal distribution in standard deviations


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
    fusion: str = "mean"
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
        fusion = settings.get("fusion", "mean")
        if fusion not in ("mean", "median"):
            raise ValueError("fusion must be 'mean' or 'median'")
        visual, text = unit_rows(visual), unit_rows(text)
        model = cls(visual=visual, text=text, captions=[e.description for e in memory_events],
                    weights=weights, k=k, centering=centering, fusion=fusion)
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
        if weights["judge"]:
            held_out["judge"] = np.asarray(judge_scores, dtype=float)
        floors = {"visual": 1e-3, "text": 1e-3, "judge": .5}
        model.scales = {name: RobustScale.fit(values, floors[name]) for name, values in held_out.items()}
        fused = model._fuse({name: model.scales[name].z(values) for name, values in held_out.items()})
        model.scales["fused"] = RobustScale.fit(fused, 1e-3)
        model.calibration = {
            "held_out_windows": len(memory_events),
            "guard_seconds": float(settings.get("calibration_guard_seconds", 4.0)),
            **{f"{name}_median": s.center for name, s in model.scales.items()},
            **{f"{name}_spread": s.scale for name, s in model.scales.items()},
            "observation_max_z": float(model.scales["fused"].z(fused).max()),
        }
        return model

    def _fuse(self, zs):
        if self.fusion == "median":
            # Needs agreement: with visual, caption and judge terms, one alone cannot alert.
            active = [zs[name] for name in zs if self.weights[name] > 0]
            return np.median(np.stack(active), axis=0)
        total = sum(self.weights[name] for name in zs)
        return sum(self.weights[name] * zs[name] for name in zs) / total

    def _prepare(self, vectors, mean):
        vectors = unit_rows(np.atleast_2d(vectors))
        return unit_rows(vectors - mean) if self.centering else vectors

    def score(self, visual, text, judge=None):
        """Score test windows; returns per-window arrays and the nearest memory caption."""
        visual_d, nearest = knn_distance(self._prepare(visual, self.visual_mean), self.visual, self.k)
        text_d, _ = knn_distance(self._prepare(text, self.text_mean), self.text, self.k)
        zs = {"visual": self.scales["visual"].z(visual_d), "text": self.scales["text"].z(text_d)}
        if self.weights["judge"]:
            if judge is None:
                raise ValueError("This memory was calibrated with judge ratings; pass them")
            zs["judge"] = self.scales["judge"].z(judge)
        fused = self.scales["fused"].z(self._fuse(zs))
        return {"visual_distance": visual_d, "text_distance": text_d,
                **{f"{name}_z": value for name, value in zs.items()},
                "deviation": fused,
                "nearest_memory_caption": [self.captions[i] for i in nearest]}
