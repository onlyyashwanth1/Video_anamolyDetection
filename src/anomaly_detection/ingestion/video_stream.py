"""
Step 1: Live Input Stream
--------------------------
Paper reference: "LIVE INPUT STREAM (Continuous Multi-Domain Video Source)"

Reads frames from a video file or webcam and yields them one at a time as
Frame objects. This is the entry point of the pipeline - it doesn't know
or care what's actually in the video (a hallway, a warehouse, a kitchen).

Design note: we don't process every single frame. A video file might have
30 frames per second, but for anomaly detection that's overkill - most
consecutive frames look almost identical. We sample down to a target
frame rate (e.g. 1 frame/sec) to cut compute cost dramatically.
"""

from typing import Iterator, Union
import cv2

from anomaly_detection.utils.types import Frame


class VideoStream:
    def __init__(self, source: Union[str, int] = 0, target_fps: float = 1.0):
        """
        source: path to a video file, OR an integer (0 = default webcam)
        target_fps: how many frames per second we actually want to process
        """
        self.source = source
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = max(1, round(self.native_fps / self.target_fps))

    def frames(self) -> Iterator[Frame]:
        """Yields sampled Frame objects, skipping near-duplicate frames."""
        frame_idx = 0
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break  # end of video, or stream disconnected

            if frame_idx % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield Frame(
                    frame_number=frame_idx,
                    timestamp_sec=frame_idx / self.native_fps,
                    image=frame_rgb,
                )

            frame_idx += 1

        self.cap.release()

    def close(self):
        self.cap.release()
