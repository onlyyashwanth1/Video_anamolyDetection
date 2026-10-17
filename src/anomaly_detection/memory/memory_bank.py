"""
Step 7: Dynamic Textual Memory Bank
------------------------------------------------------------------
Paper reference: "DYNAMIC TEXTUAL MEMORY BANK (Auto-Refreshes & Appends
New Norms)" plus the "familiarity counter" mechanism for auto-promoting
recurring patterns into "normal" over time.

Holds the current list of "normal" MemoryEntry objects (built from
familiarisation), answers "what's the closest match to this new event?"
(used by inference/adaptive_inference.py), and tracks recurring-but-
unmatched descriptions so they can eventually be promoted into permanent
memory entries without ever retraining any model.
"""

import pickle
from typing import List, Optional, Tuple
import numpy as np

from anomaly_detection.utils.types import MemoryEntry


class MemoryBank:
    def __init__(self, initial_entries: Optional[List[MemoryEntry]] = None):
        self.entries: List[MemoryEntry] = list(initial_entries) if initial_entries else []
        # Tracks unmatched descriptions seen so far, for the familiarity counter:
        self._unmatched_counts: dict = {}
        self._unmatched_embeddings: dict = {}

    def save(self, path: str) -> None:
        """Persist the memory bank to disk via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        """Load a previously saved memory bank from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_index(self):
        """Stack all entry embeddings into a matrix for fast vectorized lookup."""
        if not hasattr(self, '_emb_matrix') or self._emb_matrix is None or len(self._emb_matrix) != len(self.entries):
            if self.entries:
                self._emb_matrix = np.stack([e.embedding for e in self.entries])  # (N, D)
            else:
                self._emb_matrix = None

    def best_match(self, embedding: np.ndarray) -> Tuple[Optional[MemoryEntry], float]:
        """
        Returns (closest MemoryEntry, similarity score). If the bank is
        empty, returns (None, -1.0) - the caller should treat this as
        "everything is anomalous" since there's nothing to compare against yet.
        """
        if not self.entries:
            return None, -1.0

        self._build_index()
        sims = self._emb_matrix @ embedding  # (N,)
        best_idx = int(np.argmax(sims))
        return self.entries[best_idx], float(sims[best_idx])

    def top_k_match(self, embedding: np.ndarray, k: int = 10) -> Tuple[Optional[MemoryEntry], float, float]:
        """
        Returns (best_entry, best_similarity, top_k_avg_similarity).
        top_k_avg is the mean similarity of the K nearest entries.
        Anomalous events will have a lower top-K average because they
        won't cluster near many normal entries.
        """
        if not self.entries:
            return None, -1.0, -1.0

        self._build_index()
        sims = self._emb_matrix @ embedding  # (N,)
        actual_k = min(k, len(sims))
        top_k_indices = np.argpartition(sims, -actual_k)[-actual_k:]
        top_k_sims = sims[top_k_indices]
        best_local = int(np.argmax(top_k_sims))
        best_idx = top_k_indices[best_local]

        return self.entries[best_idx], float(sims[best_idx]), float(np.mean(top_k_sims))

    def log_unmatched(self, description: str, embedding: np.ndarray) -> None:
        """Records an event that didn't match anything, for try_promote() to check later."""
        self._unmatched_counts[description] = self._unmatched_counts.get(description, 0) + 1
        self._unmatched_embeddings[description] = embedding  # keep latest embedding seen for this description

    def try_promote(self, description: str, repeat_threshold: int) -> bool:
        """
        If `description` has now recurred >= repeat_threshold times (via
        log_unmatched), promotes it into a permanent MemoryEntry and
        returns True. Otherwise returns False and does nothing.
        """
        count = self._unmatched_counts.get(description, 0)
        if count < repeat_threshold:
            return False

        embedding = self._unmatched_embeddings[description]
        self.entries.append(MemoryEntry(text=description, embedding=embedding, occurrence_count=count))

        # Clean up - it's promoted now, no longer "unmatched"
        del self._unmatched_counts[description]
        del self._unmatched_embeddings[description]
        return True

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        lines = "\n".join(f"  - {e.text} (seen {e.occurrence_count}x)" for e in self.entries)
        return f"MemoryBank({len(self.entries)} entries):\n{lines}"
