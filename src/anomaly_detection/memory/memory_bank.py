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

from typing import List, Optional, Tuple
import numpy as np

from anomaly_detection.utils.types import MemoryEntry


class MemoryBank:
    def __init__(self, initial_entries: Optional[List[MemoryEntry]] = None):
        self.entries: List[MemoryEntry] = list(initial_entries) if initial_entries else []
        # Tracks unmatched descriptions seen so far, for the familiarity counter:
        self._unmatched_counts: dict = {}
        self._unmatched_embeddings: dict = {}

    def best_match(self, embedding: np.ndarray) -> Tuple[Optional[MemoryEntry], float]:
        """
        Returns (closest MemoryEntry, similarity score). If the bank is
        empty, returns (None, -1.0) - the caller should treat this as
        "everything is anomalous" since there's nothing to compare against yet.
        """
        if not self.entries:
            return None, -1.0

        best_entry, best_score = None, -1.0
        for entry in self.entries:
            score = float(np.dot(embedding, entry.embedding))  # both are normalized -> dot = cosine similarity
            if score > best_score:
                best_entry, best_score = entry, score

        return best_entry, best_score

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
