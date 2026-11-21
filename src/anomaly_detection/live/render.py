"""Burn the subtitle cues into a copy of the video with OpenCV (no ffmpeg needed).

The output has no audio track. Use the .srt file with the original video
in VLC or mpv when audio matters.
"""

import bisect
import textwrap

import cv2

from anomaly_detection.live.subtitles import ANOMALY, NORMAL, OBSERVE

BGR = {OBSERVE: (255, 203, 158), NORMAL: (0, 220, 120), ANOMALY: (59, 59, 255)}


def _draw_lines(frame, lines, color, scale, thickness):
    height, width = frame.shape[:2]
    step = int(32 * scale) + 8
    top = height - step * len(lines) - 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top - 8), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, .55, frame, .45, 0, frame)
    for i, (text, line_color) in enumerate(lines):
        y = top + step * i + int(24 * scale)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, line_color or color,
                    thickness, cv2.LINE_AA)


def render(video_path, cues, output_path, threshold, observe_until):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output_path}")
    scale = max(.45, min(1.0, width / 1280))
    thickness = 2 if width >= 960 else 1
    chars = max(30, int(width / (19 * scale)))
    starts = [cue.start for cue in cues]
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time = index / fps
            position = bisect.bisect_right(starts, time) - 1
            if position >= 0 and time < cues[position].end:
                cue = cues[position]
                if cue.label == OBSERVE:
                    head = f"OBSERVING: learning normal (until {observe_until:.0f}s)"
                else:
                    head = f"{'ANOMALY' if cue.label == ANOMALY else 'NORMAL'}   score {cue.score:.2f} / alert above {threshold:.2f}"
                body = textwrap.wrap(cue.caption or "", chars)[:3]
                if cue.explanation:
                    body += textwrap.wrap("Why: " + cue.explanation, chars)[:2]
                color = BGR[cue.label]
                _draw_lines(frame, [(head, color)] + [(line, (255, 255, 255)) for line in body],
                            color, scale, thickness)
                if cue.label == ANOMALY:
                    cv2.rectangle(frame, (2, 2), (width - 3, height - 3), color, 6)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()
    return output_path
