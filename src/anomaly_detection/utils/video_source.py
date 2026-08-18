"""Video source resolution helpers.

Accepts normal local video paths as well as public Google Drive sharing URLs.
Google Drive URLs are downloaded to a temporary local file before OpenCV opens
 them. The ML pipeline itself remains unchanged and continues to consume a
local video path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse


Source = Union[str, int]


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
