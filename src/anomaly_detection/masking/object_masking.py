"""
Step 4: Spatial Object-Centric Masking
------------------------------------------------------------------
Paper reference: "Encode 'Cognitive Filters' via Spatial Object-Centric Masking"
Isolates people and behaviorally relevant objects using YOLO-World (open-vocabulary object detector)
bound to ByteTrack (cross-frame tracker), applying a background blurring transform so background
environmental shifts (weather, lighting, shadows) do not generate false positive alerts.
"""

import cv2
import numpy as np
from typing import List, Optional
from anomaly_detection.utils.types import Frame


class SpatialObjectMasker:
    def __init__(self, model_name: str = "yolov8s-worldv2.pt", custom_classes: Optional[List[str]] = None, blur_kernel: tuple = (51, 51)):
        self.blur_kernel = blur_kernel
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            if custom_classes:
                self.model.set_classes(custom_classes)
            self.enabled = True
        except Exception as e:
            print(f"Warning: Could not initialize YOLO-World ({e}). Falling back to unmasked frames.")
            self.enabled = False

    def mask_frame(self, frame: Frame) -> Frame:
        """
        Detects & tracks foreground objects on frame.image via YOLO-World + ByteTrack,
        creates a foreground binary mask, blurs background pixels outside bounding boxes,
        and returns a new Frame object with background noise suppressed.
        """
        if not self.enabled or frame.image is None:
            return frame

        try:
            results = self.model.track(frame.image, persist=True, tracker="bytetrack.yaml", verbose=False)
            if not results or len(results) == 0:
                return frame

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return frame

            h, w, c = frame.image.shape
            mask = np.zeros((h, w), dtype=np.uint8)

            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255

            blurred_bg = cv2.GaussianBlur(frame.image, self.blur_kernel, 0)
            masked_img = frame.image.copy()
            masked_img[mask == 0] = blurred_bg[mask == 0]

            return Frame(
                frame_number=frame.frame_number,
                timestamp_sec=frame.timestamp_sec,
                image=masked_img
            )
        except Exception as e:
            return frame


_masker_instance = None


def mask_frame(frame: Frame) -> Frame:
    """
    Convenience function matching planned interface: runs YOLO-World + ByteTrack
    background blurring transform on input Frame.
    """
    global _masker_instance
    if _masker_instance is None:
        _masker_instance = SpatialObjectMasker()
    return _masker_instance.mask_frame(frame)

