"""
Sanity test for Step 3 (segmentation/temporal_decomposition.py) that does
NOT require downloading CLIP - it uses fake, hand-crafted fingerprints
wrapped in real Frame/EncodedFrame objects, so you can verify the grouping
logic works before wiring up the real model.

Run with:  python tests/test_temporal_decomposition.py
       or:  pytest tests/
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anomaly_detection.utils.types import Frame, EncodedFrame
from anomaly_detection.segmentation.temporal_decomposition import build_event_tree


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def make_encoded_frames(vectors):
    encoded = []
    for i, v in enumerate(vectors):
        frame = Frame(frame_number=i, timestamp_sec=float(i), image=np.zeros((2, 2, 3)))
        encoded.append(EncodedFrame(frame=frame, embedding=unit(v)))
    return encoded


def test_basic_split():
    # Simulates: 3 "empty hallway" frames, 3 "person walking" frames, 2 "empty" frames
    encoded_frames = make_encoded_frames([
        [1, 0], [0.95, 0.05], [0.97, 0.03],   # empty
        [0, 1], [0.05, 0.95], [0.02, 0.98],   # walking
        [1, 0], [0.96, 0.04],                  # empty again
    ])

    tree = build_event_tree(encoded_frames, coarse_threshold=0.7, fine_threshold=0.999)

    assert len(tree) == 3, f"expected 3 top-level events, got {len(tree)}"
    print(f"PASSED: got {len(tree)} top-level events as expected")
    for chunk in tree:
        print(" ", chunk)


def test_fine_grained_subsplits():
    # A single coarse "walking" event with two internal sub-phases
    encoded_frames = make_encoded_frames([
        [0, 1], [0.1, 0.99],       # sub-phase A: walking in
        [0.9, 0.4], [0.95, 0.3],   # sub-phase B: turning to exit
    ])

    tree = build_event_tree(encoded_frames, coarse_threshold=0.0, fine_threshold=0.8)

    assert len(tree) == 1, f"expected 1 coarse event, got {len(tree)}"
    assert len(tree[0].children) >= 2, "expected sub-events to be detected"
    print(f"PASSED: 1 coarse event containing {len(tree[0].children)} sub-events")
    for child in tree[0].children:
        print("   ", child)


if __name__ == "__main__":
    test_basic_split()
    test_fine_grained_subsplits()
    print("\nAll tests passed.")
