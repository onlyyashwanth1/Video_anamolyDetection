"""
Step 3: Automated Temporal Decomposition
-------------------------------------------
Paper reference: "AUTOMATED TEMPORAL DECOMPOSITION (Event Boundaries &
Video Tree)" - built from a GEBD (Generic Event Boundary Detection) model
feeding a Hierarchical Granularity-Aware Tree (HGTree).

Groups a sequence of encoded frames into "events" (chunks), organized as
a two-level tree:
  - Coarse level: big, obvious scene changes (empty -> person walks -> empty)
  - Fine level:   sub-actions within a coarse event (enters -> walks -> exits)

Mechanism: walk through consecutive frames and measure cosine similarity
between their CLIP fingerprints. A small change means "still the same
event" - keep grouping. A big drop in similarity means "something
changed" - start a new event.

IMPLEMENTATION NOTE: this is a simplified similarity-drop heuristic
standing in for a dedicated trained GEBD model, since no off-the-shelf
pretrained GEBD checkpoint is a drop-in pip package - real
implementations are research code with fiddly setup. This heuristic is
fully functional and can be swapped out later (see build_event_tree's
signature - it only needs a list of EncodedFrame, so a real GEBD model
could replace _split_by_similarity without touching any other module).
"""

from typing import List
import numpy as np

from anomaly_detection.utils.types import EncodedFrame, EventChunk


def _split_by_similarity(embeddings: List[np.ndarray], indices: List[int],
                          threshold: float) -> List[List[int]]:
    """
    Core mechanism: walk through consecutive frames (within `indices`),
    compare fingerprints, and start a new group whenever similarity drops
    below `threshold`. Returns a list of groups, each a list of frame indices.
    """
    if not indices:
        return []

    groups = [[indices[0]]]
    for i in range(1, len(indices)):
        prev_idx, curr_idx = indices[i - 1], indices[i]
        sim = float(np.dot(embeddings[prev_idx], embeddings[curr_idx]))

        if sim < threshold:
            groups.append([curr_idx])       # big jump -> start a new group
        else:
            groups[-1].append(curr_idx)     # small change -> keep same group

    return groups


def build_event_tree(encoded_frames: List[EncodedFrame],
                      coarse_threshold: float = 0.80,
                      fine_threshold: float = 0.92) -> List[EventChunk]:
    """
    Builds a two-level hierarchy of EventChunks from a list of EncodedFrame.

    coarse_threshold: similarity below this = a NEW top-level event.
                       Lower value -> only splits on big, obvious changes.
    fine_threshold:   similarity below this = a NEW sub-event within a
                       coarse event. Higher value -> more sensitive to
                       smaller internal changes.
    """
    embeddings = [ef.embedding for ef in encoded_frames]
    all_indices = list(range(len(encoded_frames)))
    coarse_groups = _split_by_similarity(embeddings, all_indices, coarse_threshold)

    tree: List[EventChunk] = []
    for group in coarse_groups:
        first, last = encoded_frames[group[0]].frame, encoded_frames[group[-1]].frame
        coarse_chunk = EventChunk(
            start_frame=first.frame_number,
            end_frame=last.frame_number,
            start_time=first.timestamp_sec,
            end_time=last.timestamp_sec,
            frame_indices=group,
        )

        # Only worth splitting further if the coarse event spans a few frames
        if len(group) >= 3:
            fine_groups = _split_by_similarity(embeddings, group, fine_threshold)
            if len(fine_groups) > 1:
                for fg in fine_groups:
                    f_first, f_last = encoded_frames[fg[0]].frame, encoded_frames[fg[-1]].frame
                    coarse_chunk.children.append(EventChunk(
                        start_frame=f_first.frame_number,
                        end_frame=f_last.frame_number,
                        start_time=f_first.timestamp_sec,
                        end_time=f_last.timestamp_sec,
                        frame_indices=fg,
                    ))

        tree.append(coarse_chunk)

    return tree
