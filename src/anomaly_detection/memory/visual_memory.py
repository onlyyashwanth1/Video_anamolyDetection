"""Normal visual exemplars alongside the existing textual memory.

Calibration uses separate normal videos, never the reference bank itself or
test labels. Feature generation is unchanged, so existing video caches apply.
"""

import numpy as np

from anomaly_detection.memory.memory_bank import MemoryBank


def unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if (vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError("Expected nonempty finite normalized embedding rows")
    return vectors


def fit_calibration(distances):
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires nonempty finite normal distances")
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    return {"center": float(median), "scale": float(max(q75 - q25, 1e-4)),
            "method": "normal_median_iqr_sigmoid_v1", "samples": len(values)}


def calibrate(distances, parameters):
    center, scale = float(parameters["center"]), float(parameters["scale"])
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid normal calibration")
    return 1 / (1 + np.exp(-np.clip((np.asarray(distances) - center) / scale, -60, 60)))


class NeighborIndex:
    """Exact batched cosine retrieval; never materializes the entire Q x N matrix."""

    def __init__(self, vectors, device="cpu", query_batch=128, reference_batch=32768):
        self.vectors = unit_rows(vectors)
        self.device = str(device)
        self.query_batch, self.reference_batch = int(query_batch), int(reference_batch)
        if min(self.query_batch, self.reference_batch) < 1:
            raise ValueError("Search batch sizes must be positive")
        self.tensor = None
        if self.device != "cpu":
            import torch
            # Exact float32 retrieval is also used when preparing tuning features.
            self.tensor = torch.as_tensor(self.vectors, device=self.device)

    def distances(self, queries, k=10):
        queries = unit_rows(queries)
        if queries.shape[1] != self.vectors.shape[1] or k < 1:
            raise ValueError("Invalid query dimension or neighbor count")
        k = min(int(k), len(self.vectors))
        outputs = []
        for start in range(0, len(queries), self.query_batch):
            batch = queries[start:start + self.query_batch]
            if self.tensor is not None:
                import torch
                with torch.inference_mode():
                    query = torch.as_tensor(batch, device=self.device)
                    best = torch.empty((len(batch), 0), device=self.device)
                    for offset in range(0, len(self.vectors), self.reference_batch):
                        sims = query @ self.tensor[offset:offset + self.reference_batch].T
                        nearest = torch.topk(sims, min(k, sims.shape[1]), dim=1).values
                        best = torch.topk(torch.cat((best, nearest), dim=1),
                                          min(k, best.shape[1] + nearest.shape[1]), dim=1).values
                    distance = 1 - (.6 * best.mean(1) + .4 * best.max(1).values)
                    outputs.append(distance.cpu().numpy())
            else:
                best = np.empty((len(batch), 0), dtype=np.float32)
                for offset in range(0, len(self.vectors), self.reference_batch):
                    sims = batch @ self.vectors[offset:offset + self.reference_batch].T
                    take = min(k, sims.shape[1])
                    nearest = np.partition(sims, -take, axis=1)[:, -take:]
                    candidates = np.concatenate((best, nearest), axis=1)
                    take = min(k, candidates.shape[1])
                    best = np.partition(candidates, -take, axis=1)[:, -take:]
                outputs.append(1 - (.6 * best.mean(1) + .4 * best.max(1)))
        return np.concatenate(outputs).astype(float)

    def closest(self, vector):
        if self.tensor is not None:
            import torch
            with torch.inference_mode():
                query = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=self.device)
                return int(torch.argmax(self.tensor @ query).item())
        return int(np.argmax(self.vectors @ vector))


class VisualTextMemoryBank(MemoryBank):
    def __init__(self, entries, visual_embeddings, visual_text_indices, visual_sources,
                 visual_starts, calibration, device="cpu", top_k=10):
        super().__init__(entries)
        self.visual_embeddings = unit_rows(visual_embeddings)
        self.visual_text_indices = np.asarray(visual_text_indices, dtype=np.int64)
        self.visual_sources = np.asarray(visual_sources, dtype=str)
        self.visual_starts = np.asarray(visual_starts, dtype=np.int64)
        self.calibration = calibration
        if (not entries or any(len(a) != len(self.visual_embeddings) for a in
                              (self.visual_text_indices, self.visual_sources, self.visual_starts))
                or np.any(self.visual_text_indices < 0)
                or np.any(self.visual_text_indices >= len(entries))
                or np.any(self.visual_starts < 0)):
            raise ValueError("Visual exemplars must have aligned caption/source references")
        for key in ("visual", "text"):
            calibrate([0.0], calibration[key])
        self.search_device = device
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        self._indices = None

    def distances(self, image_vectors, text_vectors, k=10):
        if self._indices is None:
            self._indices = (NeighborIndex(self.visual_embeddings, self.search_device),
                             NeighborIndex(np.stack([e.embedding for e in self.entries]), self.search_device))
        return (self._indices[0].distances(image_vectors, k),
                self._indices[1].distances(text_vectors, k))

    def score_memory(self, image_vector, text_vector, image_weight=.5, k=10):
        if not np.isfinite(image_weight) or not 0 <= image_weight <= 1:
            raise ValueError("Memory image_weight must be finite and in [0, 1]")
        image_raw, text_raw = self.distances(np.asarray(image_vector)[None], np.asarray(text_vector)[None], k)
        image_score = float(calibrate(image_raw, self.calibration["visual"])[0])
        text_score = float(calibrate(text_raw, self.calibration["text"])[0])
        return {"memory_image_raw": float(image_raw[0]), "memory_text_raw": float(text_raw[0]),
                "memory_image_score": image_score, "memory_text_score": text_score,
                "memory_score": image_weight * image_score + (1 - image_weight) * text_score}

    def closest_visual(self, vector):
        index = (self._indices[0] if self._indices is not None else
                 NeighborIndex(self.visual_embeddings, self.search_device)).closest(vector)
        return self.entries[self.visual_text_indices[index]]
