"""Turn overlapping scored windows into readable, non-overlapping subtitles.

Windows are 4 s long and start every second, so they overlap. The timeline
is cut at window starts. Each test segment gets the mean deviation of the
windows that cover it. A cue then runs for at most ``cue_seconds`` and is
cut early whenever the label changes, so an anomaly cue starts at onset.
A cue's caption is the caption of the window that starts with the cue.
"""

from dataclasses import dataclass, field

OBSERVE, NORMAL, ANOMALY = "OBSERVING", "NORMAL", "ANOMALY"
COLORS = {OBSERVE: "#9ecbff", NORMAL: "#7CFC00", ANOMALY: "#ff3b3b"}


def _plain(text):
    """Players parse <...> as tags in SRT; keep caption text literal."""
    return text.replace("<", "(").replace(">", ")")


@dataclass
class Cue:
    start: float
    end: float
    label: str
    caption: str
    deviation: float = None
    window_index: int = None
    explanation: str = None
    extra: dict = field(default_factory=dict)


def build_segments(memory, test, test_deviation, duration):
    """Return [(start, end, phase, window, deviation)] covering the whole video."""
    segments = []
    for i, window in enumerate(memory):
        end = memory[i + 1].start_time if i + 1 < len(memory) else test[0].start_time
        segments.append((window.start_time, end, OBSERVE, ("memory", i), None))
    for j, window in enumerate(test):
        end = test[j + 1].start_time if j + 1 < len(test) else max(window.end_time, duration)
        middle = (window.start_time + end) / 2
        covering = [test_deviation[n] for n, w in enumerate(test) if w.start_time <= middle < w.end_time]
        segments.append((window.start_time, end, "TEST", ("test", j),
                         float(sum(covering) / len(covering)) if covering else float(test_deviation[j])))
    return [s for s in segments if s[1] > s[0]]


def build_cues(segments, captions, threshold, cue_seconds=4.0):
    """Group segments into cues of one label; ``captions`` maps window keys to text."""
    cues = []
    for start, end, phase, key, deviation in segments:
        label = OBSERVE if phase == OBSERVE else (ANOMALY if deviation > threshold else NORMAL)
        last = cues[-1] if cues else None
        if last and last.label == label and end - last.start <= cue_seconds + 1e-6:
            last.end = end
            if deviation is not None:
                last.deviation = max(last.deviation, deviation)
            continue
        cues.append(Cue(start, end, label, captions[key], deviation,
                        key[1] if key[0] == "test" else None))
    return cues


def timestamp(seconds):
    milliseconds = int(round(max(seconds, 0) * 1000))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def cue_lines(cue, threshold, observe_until, colors=True):
    if cue.label == OBSERVE:
        tag = f"[OBSERVING - learning normal, until {observe_until:.0f}s]"
    elif cue.label == NORMAL:
        tag = f"[NORMAL] deviation {cue.deviation:.1f} (alert at {threshold:.1f})"
    else:
        tag = f"[!! ANOMALY !!] deviation {cue.deviation:.1f} (alert at {threshold:.1f})"
    caption = cue.caption or "(no caption)"
    lines = [tag, caption]
    if cue.explanation:
        lines.append(f"Why: {cue.explanation}")
    if not colors:
        return [_plain(line) for line in lines]
    color = COLORS[cue.label]
    tag_html = _plain(tag)
    if cue.label == ANOMALY:
        tag_html = f"<b>{tag_html}</b>"
    lines = [f'<font color="{color}">{tag_html}</font>'] + [_plain(line) for line in lines[1:]]
    if cue.label == ANOMALY:
        lines[1:] = [f'<font color="{color}">{line}</font>' for line in lines[1:]]
    return lines


def write_srt(cues, path, threshold, observe_until, colors=True):
    with open(path, "w", encoding="utf-8") as handle:
        for number, cue in enumerate(cues, 1):
            handle.write(f"{number}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n")
            handle.write("\n".join(cue_lines(cue, threshold, observe_until, colors)) + "\n\n")
