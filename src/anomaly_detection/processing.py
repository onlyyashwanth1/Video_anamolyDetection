"""Shared video-to-event processing for demos, memory construction and evaluation."""

from dataclasses import dataclass

from anomaly_detection.encoding.multi_scale_encoder import MultiScaleClipEncoder
from anomaly_detection.familiarisation.domain_familiarisation import caption_event, caption_windows
from anomaly_detection.ingestion.video_stream import VideoStream
from anomaly_detection.masking.object_masking import SpatialObjectMasker
from anomaly_detection.segmentation.temporal_decomposition import build_event_tree
from anomaly_detection.segmentation.sliding_windows import build_sliding_windows


@dataclass
class ProcessedVideo:
    events: list
    frames: list
    encoded_frames: list
    fps: float
    frame_count: int


class VideoEventProcessor:
    """Reuse pretrained models, but reset tracking at each video boundary.

    Mean-pooled CLIP features preserve image/text alignment. The experimental
    randomly initialized temporal transformer is intentionally not applied.
    """

    def __init__(self, config, strict=False):
        self.config = config
        self.strict = strict
        self.encoder = MultiScaleClipEncoder(model_name=config["encoder"]["model_name"])
        self.masker = SpatialObjectMasker(strict=strict)

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
        self.masker.reset_tracking()
        encoded = [self.encoder.encode_frame(self.masker.mask_frame(f)) for f in frames]
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
