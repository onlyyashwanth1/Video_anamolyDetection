"""Shared perception: video -> sampled frames -> CLIP embeddings -> windows/events -> captions.

Every architecture and the evaluation cache use this one path, so the same
video always produces the same windows, embeddings and captions.
"""

from dataclasses import dataclass

from anomaly_detection.perception.captioner import caption_event, caption_windows
from anomaly_detection.perception.clip_encoder import MultiScaleClipEncoder
from anomaly_detection.perception.events import build_event_tree
from anomaly_detection.perception.video import VideoStream
from anomaly_detection.perception.windows import build_sliding_windows


@dataclass
class ProcessedVideo:
    events: list
    frames: list
    encoded_frames: list
    fps: float
    frame_count: int


class VideoEventProcessor:
    """Load the frozen encoders once and reuse them for every video.

    Window and event embeddings are the mean of their frames' multi-scale CLIP
    embeddings, renormalized. Frames are not masked.
    """

    def __init__(self, config, strict=False):
        self.config = config
        self.strict = strict
        self.encoder = MultiScaleClipEncoder(model_name=config["encoder"]["model_name"])

    def process(self, source, caption=True):
        import cv2

        stream = VideoStream(source, target_fps=self.config["video"]["target_fps"])
        native_fps = stream.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.strict and (native_fps <= 0 or frame_count <= 0):
            stream.close()
            raise ValueError(f"Invalid video metadata: {source}")
        try:
            frames = list(stream.frames())
        finally:
            stream.close()
        if not frames:
            raise ValueError(f"No frames decoded: {source}")
        if self.strict:
            expected = (frame_count + stream.frame_interval - 1) // stream.frame_interval
            if len(frames) != expected:
                raise ValueError(f"Incomplete video decode: {source} ({len(frames)}/{expected} samples)")
        encoded = [self.encoder.encode_frame(f) for f in frames]
        temporal = self.config.get("temporal", {})
        mode = temporal.get("mode", "events")
        if mode == "sliding_windows":
            tree = build_sliding_windows(encoded, frame_count, stream.native_fps,
                                         temporal.get("window_seconds", 4.0),
                                         temporal.get("stride_seconds", 1.0))
        elif mode == "events":
            tree = build_event_tree(encoded, **self.config["segmentation"])
        else:
            raise ValueError(f"Unknown temporal mode: {mode}")
        if caption:
            self.caption_events(tree, frames)
        return ProcessedVideo(tree, frames, encoded, stream.native_fps, frame_count)

    def caption_events(self, events, frames):
        settings = self.config.get("captioning", {})
        mode = settings.get("mode", "middle_frame")
        if mode == "frame_captions_then_summary":
            caption_windows(events, self.encoder, frames, settings, strict=self.strict)
        elif mode == "middle_frame":
            for event in events:
                event.description = caption_event(event, self.encoder, raw_frames=frames,
                                                  strict=self.strict)
        else:
            raise ValueError(f"Unknown captioning mode: {mode}")
