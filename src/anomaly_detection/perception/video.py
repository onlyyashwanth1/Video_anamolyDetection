"""Video input: frame sampling and source resolution.

VideoStream samples a file or webcam at a target FPS. resolve_video_source
downloads a public Google Drive link to a temporary file first; local paths
and webcam indices pass through unchanged.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
from urllib.parse import urlparse

from anomaly_detection.datatypes import Frame

Source = Union[str, int]


class VideoStream:
    def __init__(self, source: Source = 0, target_fps: float = 1.0):
        """source: a video path or webcam index; target_fps: frames kept per second."""
        import cv2

        self.source = source
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = max(1, round(self.native_fps / self.target_fps))

    def frames(self) -> Iterator[Frame]:
        """Yield every frame_interval-th frame as RGB."""
        import cv2

        frame_idx = 0
        while True:
            ok, frame_bgr = self.cap.read()
            if not ok:
                break
            if frame_idx % self.frame_interval == 0:
                yield Frame(frame_number=frame_idx, timestamp_sec=frame_idx / self.native_fps,
                            image=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        self.cap.release()

    def close(self):
        self.cap.release()




def is_google_drive_url(source: str) -> bool:
    """Return True for common Google Drive sharing/download URLs."""
    try:
        parsed = urlparse(source)
    except ValueError:
        return False

    host = parsed.netloc.lower().split(":", 1)[0]
    return host == "drive.google.com" or host.endswith(".drive.google.com")


def resolve_video_source(source: Source) -> Tuple[Source, Optional[str]]:
    """Resolve a video source into something OpenCV can consume.

    Returns
    -------
    (resolved_source, temporary_path)
        For a local path or webcam integer, the source is returned unchanged
        and temporary_path is None.
        For a Google Drive URL, the public file is downloaded to a temporary
        file and both the path and cleanup path are returned.

    Notes
    -----
    Google Drive files must be accessible through a public/shared link that
    does not require an interactive Google login. Authentication for private
    Drive files is intentionally not handled here.
    """
    if isinstance(source, int):
        return source, None

    source = str(source).strip()
    if not is_google_drive_url(source):
        return source, None

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise RuntimeError(
            "Google Drive input requires the 'gdown' package. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    temp_dir = tempfile.mkdtemp(prefix="vad_gdrive_")
    output_path = os.path.join(temp_dir, "input_video")

    try:
        downloaded = gdown.download(
            url=source,
            output=output_path,
            quiet=False,
            fuzzy=True,
        )
    except Exception as exc:
        _remove_temp_dir(temp_dir)
        raise RuntimeError(
            "Could not download the Google Drive video. Make sure the link "
            "is a publicly accessible/shared file link."
        ) from exc

    if not downloaded or not os.path.isfile(downloaded):
        _remove_temp_dir(temp_dir)
        raise RuntimeError(
            "Google Drive download did not produce a video file. "
            "Check that the supplied link points to a single video file."
        )

    # gdown may append/retain an extension depending on the Drive response.
    # OpenCV uses the file contents, so no extension is required here.
    return downloaded, temp_dir


def cleanup_video_source(temporary_path: Optional[str]) -> None:
    """Delete a temporary Google Drive download after pipeline execution."""
    if temporary_path:
        _remove_temp_dir(temporary_path)


def _remove_temp_dir(path: str) -> None:
    import shutil

    shutil.rmtree(Path(path), ignore_errors=True)
