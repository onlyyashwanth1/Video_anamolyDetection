"""Memory architecture: novelty against saved normal memory (B), optionally + judge (B+J).

For each window of a test video:
  1. Caption novelty. Embed captions with MPNet, subtract centering_alpha x the
     video's mean caption vector (removes "what this scene is"), renormalize, and
     take 1 - cosine similarity to the most similar centered normal-memory caption.
  2. Image novelty. 1 - cosine similarity of the window's CLIP image embedding to
     the most similar normal-memory window (no centering).
  3. Robust z-scores against the normal memory itself: every memory window is
     scored the same way with its own video left out; z = (x - median) / (IQR / 1.349).
  4. B = z_caption + image_weight * z_image.
  5. Optional J (scoring/judge.py): B + judge_weight * z_judge.
Frames take the mean of their windows, smoothed over min(36 s, length / 4)
(temporal settings), and frame_unit() maps the result to [0, 1] with
sigmoid(score - threshold_z), so the alert threshold is 0.5.

Centering uses the whole video's mean caption and the smoothing looks ahead, so
this architecture scores complete videos (offline), not live streams.
"""

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from anomaly_detection.datatypes import AnomalyResult

LOGGER = logging.getLogger(__name__)
REFERENCE_VERSION = 1


def robust_z_params(reference):
    """Median and IQR/1.349 (std, then 1, when the IQR is zero) of finite reference values."""
    reference = np.asarray(reference, dtype=np.float64)
    finite = reference[np.isfinite(reference)]
    if not len(finite):
        raise ValueError("Robust z needs finite reference values")
    median = float(np.median(finite))
    filled = np.where(np.isfinite(reference), reference, median)
    q1, q3 = np.percentile(filled, [25, 75])
    scale = float((q3 - q1) / 1.349 or np.std(filled) or 1.0)
    return {"median": median, "scale": scale}


def robust_z(values, params):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, params["median"])
    return (values - params["median"]) / params["scale"]


def center_rows(vectors, groups, alpha):
    """Subtract alpha x each group's mean row, then renormalize rows."""
    vectors = np.asarray(vectors, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.empty_like(vectors)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        residual = vectors[rows] - alpha * vectors[rows].mean(0)
        out[rows] = residual / np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-6)
    return out.astype(np.float32)


def max_similarity(queries, memory, query_groups=None, memory_groups=None, device="cpu",
                   query_chunk=2048):
    """Highest cosine similarity of each query row to memory rows (unit vectors).

    With groups, memory rows from the query's own group are excluded.
    """
    queries = np.asarray(queries, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    exclude = query_groups is not None
    out = np.empty(len(queries), np.float32)
    if str(device) != "cpu":
        import torch
        with torch.inference_mode():
            mem = torch.as_tensor(memory, device=device)
            mg = torch.as_tensor(np.asarray(memory_groups), device=device) if exclude else None
            for start in range(0, len(queries), query_chunk):
                sims = torch.as_tensor(queries[start:start + query_chunk], device=device) @ mem.T
                if exclude:
                    qg = torch.as_tensor(np.asarray(query_groups[start:start + query_chunk]), device=device)
                    sims.masked_fill_(qg[:, None] == mg[None, :], -4.0)
                out[start:start + query_chunk] = sims.max(1).values.cpu().numpy()
        return out
    for start in range(0, len(queries), query_chunk):
        sims = queries[start:start + query_chunk] @ memory.T
        if exclude:
            qg = np.asarray(query_groups[start:start + query_chunk])
            sims[qg[:, None] == np.asarray(memory_groups)[None, :]] = -4.0
        out[start:start + query_chunk] = sims.max(1)
    return out


class NoveltyMemory:
    """Normal-memory side of B: centered caption vectors, image vectors, normal z parameters.

    Built from the saved bank's reference windows. The leave-own-video-out
    reference statistics cost O(N^2); they are cached next to the bank when
    cache_path is given.
    """

    def __init__(self, bank, text_encoder, alpha=1.0, device="cpu", cache_path=None):
        self.alpha, self.device = float(alpha), device
        captions = [bank.entries[i].text for i in bank.visual_text_indices]
        _, self.groups = np.unique(np.asarray(bank.visual_sources), return_inverse=True)
        self.entry_texts = [entry.text for entry in bank.entries]
        self.entry_text_vectors = np.asarray(text_encoder.encode(self.entry_texts), dtype=np.float32)
        lookup = {text: i for i, text in enumerate(self.entry_texts)}
        caption_vectors = self.entry_text_vectors[[lookup[c] for c in captions]]
        self.text = center_rows(caption_vectors, self.groups, self.alpha)
        self.images = np.asarray(bank.visual_embeddings, dtype=np.float32)
        self.window_entry = np.asarray(bank.visual_text_indices)
        key = self._key(bank, text_encoder)
        stored = self._load(cache_path, key)
        if stored is None:
            LOGGER.info("Scoring %d normal-memory windows against the rest of memory", len(self.images))
            text_ref = 1 - max_similarity(self.text, self.text, self.groups, self.groups, device)
            image_ref = 1 - max_similarity(self.images, self.images, self.groups, self.groups, device)
            stored = {"key": key, "text": robust_z_params(text_ref), "image": robust_z_params(image_ref)}
            if cache_path is not None:
                Path(cache_path).write_text(json.dumps(stored, indent=2) + "\n")
        self.text_params, self.image_params = stored["text"], stored["image"]

    def _key(self, bank, text_encoder):
        digest = hashlib.sha256()
        for array in (self.images, self.window_entry, np.asarray(bank.visual_sources)):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update("\n".join(self.entry_texts).encode())
        return {"version": REFERENCE_VERSION, "bank_sha256": digest.hexdigest(),
                "text_encoder": getattr(text_encoder, "name", "unknown"), "alpha": self.alpha}

    @staticmethod
    def _load(cache_path, key):
        if cache_path is None or not Path(cache_path).exists():
            return None
        stored = json.loads(Path(cache_path).read_text())
        if stored.get("key") != key:
            raise ValueError(f"{cache_path} was computed for a different bank/encoder; delete it to recompute")
        return stored


class MemoryNoveltyScorer:
    def __init__(self, settings, memory, text_encoder, judge=None):
        self.memory, self.text_encoder, self.judge = memory, text_encoder, judge
        self.image_weight = float(settings.get("image_weight", 0.25))
        self.judge_weight = float(settings.get("judge_weight", 0.0))
        self.threshold_z = float(settings.get("threshold_z", 3.0))
        self.threshold = 0.5
        for name in ("image_weight", "judge_weight", "threshold_z"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"memory_novelty.{name} must be finite")
        if self.judge_weight < 0 or self.image_weight < 0:
            raise ValueError("memory_novelty weights must be nonnegative")
        if self.judge_weight > 0 and judge is None:
            raise ValueError("memory_novelty.judge_weight > 0 needs the Gemma judge")

    def window_terms(self, captions, images):
        """B components for the windows of ONE video (rows in time order)."""
        if not captions or any(not c for c in captions):
            raise ValueError("Every window needs a caption")
        text = center_rows(self.text_encoder.encode(captions), np.zeros(len(captions), int),
                           self.memory.alpha)
        text_novelty = 1 - max_similarity(text, self.memory.text, device=self.memory.device)
        image_novelty = 1 - max_similarity(np.asarray(images, dtype=np.float32), self.memory.images,
                                           device=self.memory.device)
        z_text = robust_z(text_novelty, self.memory.text_params)
        z_image = robust_z(image_novelty, self.memory.image_params)
        terms = {"text_novelty": text_novelty, "image_novelty": image_novelty,
                 "z_text": z_text, "z_image": z_image, "B": z_text + self.image_weight * z_image}
        score = terms["B"]
        if self.judge_weight > 0:
            terms["judge_rating"], terms["z_judge"] = self.judge.score(captions)
            score = score + self.judge_weight * terms["z_judge"]
        terms["score"] = score
        return terms

    def score_video(self, events, frame_count=None, fps=None):
        terms = self.window_terms([e.description for e in events],
                                  np.stack([e.average_embedding for e in events]))
        results = []
        for i, event in enumerate(events):
            components = {k: float(v[i]) for k, v in terms.items()}
            score = components["score"]
            results.append(AnomalyResult(event=event, anomaly_score=score,
                                         is_anomaly=self.unit(score) > self.threshold,
                                         closest_match=None, component_scores=components))
        return results

    def unit(self, scores):
        """Monotonic map to [0, 1]; threshold_z maps to 0.5. AUC/AP are unchanged by it."""
        return 1 / (1 + np.exp(-np.clip(np.asarray(scores, dtype=np.float64) - self.threshold_z, -60, 60)))
