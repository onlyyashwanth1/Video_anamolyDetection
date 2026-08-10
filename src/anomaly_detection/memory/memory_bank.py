"""
Step 7: Dynamic Textual Memory Bank  [NOT YET IMPLEMENTED]
------------------------------------------------------------------
Paper reference: "DYNAMIC TEXTUAL MEMORY BANK (Auto-Refreshes & Appends
New Norms)" plus the "familiarity counter" mechanism for auto-promoting
recurring patterns into "normal" over time.

Where this fits in the pipeline:
    familiarisation (Steps 5-6) -> memory bank (THIS MODULE) initializes it.
    Then inference (Steps 8-9) reads from it on every new event.
    Unmatched-but-recurring events get logged here and, after enough
    repetitions, promoted into permanent memory entries.

Planned interface:
    class MemoryBank:
        entries: list[MemoryEntry]

        def __init__(self, initial_entries: list[MemoryEntry])
        def best_match(self, embedding: np.ndarray) -> tuple[MemoryEntry, float]
            Returns the closest MemoryEntry and its cosine similarity score.
        def log_unmatched(self, description: str, embedding: np.ndarray) -> None
            Records an event that didn't match anything, for the
            familiarity counter to track.
        def try_promote(self, description: str, repeat_threshold: int) -> bool
            Checks if `description` has recurred >= repeat_threshold times
            in the unmatched log; if so, adds it as a new permanent
            MemoryEntry and returns True.

TODO:
    - Implement in-memory storage (a dict keyed by normalized description
      text, or by embedding-cluster ID for near-duplicate descriptions)
    - Implement the repeat-count logic described in familiarisation
      (config: memory.promotion_repeat_count)
    - Consider persistence (saving/loading the bank to disk) so it
      survives restarts
"""

from typing import List, Tuple
import numpy as np

from anomaly_detection.utils.types import MemoryEntry


class MemoryBank:
    def __init__(self, initial_entries: List[MemoryEntry] = None):
        raise NotImplementedError(
            "MemoryBank is not implemented yet. "
            "See this module's docstring for the planned interface and TODOs."
        )

    def best_match(self, embedding: np.ndarray) -> Tuple[MemoryEntry, float]:
        raise NotImplementedError

    def log_unmatched(self, description: str, embedding: np.ndarray) -> None:
        raise NotImplementedError

    def try_promote(self, description: str, repeat_threshold: int) -> bool:
        raise NotImplementedError
