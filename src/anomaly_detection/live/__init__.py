"""Single-video "observe, then detect" mode.

The first part of a video (40% by default) is watched without alerts and
stored as this scene's normal memory. Every later window is scored only
against that memory, and the result is written as an SRT subtitle track.
"""
