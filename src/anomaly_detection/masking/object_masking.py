"""
Step 4: Spatial Object-Centric Masking  [NOT YET IMPLEMENTED]
------------------------------------------------------------------
Paper reference: "Encode 'Cognitive Filters' via Spatial Object-Centric
Masking" - isolates people/objects using YOLO-World (open-vocabulary
object detector) + ByteTrack (cross-frame tracker), and blurs/masks the
background, so environmental changes (weather, lighting, seasonal decor)
don't get mistaken for anomalies.

Where this fits in the pipeline:
    ingestion (Step 1) -> masking (Step 4, THIS MODULE) -> encoding (Step 2)
    i.e. mask the raw frame BEFORE it gets fingerprinted by CLIP, so the
    fingerprint mostly reflects foreground activity, not background noise.

Planned interface:
    mask_frame(frame: Frame) -> Frame
        Runs YOLO-World detection + ByteTrack tracking on frame.image,
        returns a new Frame whose .image has the background blurred/masked
        and detected foreground objects left untouched.

TODO:
    - pip install ultralytics (or the specific YOLO-World package) and load
      an open-vocabulary checkpoint
    - Integrate a ByteTrack implementation for cross-frame object ID tracking
    - Implement background blur/mask compositing (e.g. cv2.GaussianBlur
      outside detected bounding boxes)
"""

from anomaly_detection.utils.types import Frame


def mask_frame(frame: Frame) -> Frame:
    raise NotImplementedError(
        "Object masking (YOLO-World + ByteTrack) is not implemented yet. "
        "See this module's docstring for the planned interface and TODOs."
    )
