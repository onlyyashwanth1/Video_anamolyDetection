# Live mode: observe 40%, then detect (branch `live-memory-40`)

This mode works on **one video at a time**, as a camera would be used. Each
video sets up its own normal memory.

1. **Observe (first 40%).** The 4 s windows (1 s stride) are captioned by Gemma and
   stored in memory: an unmasked CLIP image embedding plus an MPNet caption embedding.
   No alerts are raised here.
2. **Freeze and calibrate.** Each observation window is compared with memory, leaving out
   itself and every window within 4 s of it. These held-out distances show how far
   ordinary moments of *this* scene usually are from memory.
3. **Test (remaining 60%).** Each new window is captioned and compared with the frozen
   memory. Its *deviation* is how many normal spreads (median / IQR) it sits above that
   held-out level. It combines visual novelty, caption novelty and, by default, a
   memory-aware Gemma judge (the B+J recipe). The judge rates the caption 0–9 after seeing
   the 8 most similar normal captions from memory. **Deviation > 3.0 = ANOMALY.**
4. **Subtitles.** The SRT covers the whole video:
   - `[OBSERVING - learning normal, until 43s]` + caption (blue)
   - `[NORMAL] deviation 0.8 (alert > 3.0)` + caption (green)
   - `[!! ANOMALY !!] deviation 5.1 (alert > 3.0)` + caption of what happens (red)

   A cue lasts at most 4 s. A new cue starts as soon as the label changes, so the anomaly
   caption appears at onset. The caption shown is from the window starting at that moment.

No saved reference bank or anomaly prompt list is used, and no test window ever enters memory.
Windows that straddle the 40% boundary are not used for either memory or testing.

## Run

```bash
cd /home/atripathi/prudhvi/Video_anamolyDetection_live
export PATH=$HOME/ollama/bin:$PATH
CUDA_VISIBLE_DEVICES=0 ../Video_anamolyDetection/.venv/bin/python scripts/run_live_memory.py \
    --video /path/to/video.mp4 --render
```

Pick a free GPU with `nvidia-smi`. CLIP, MPNet and one Ollama Gemma server (~35 GB) all run
on that GPU. Ollama listens on port 11600 (`--port` to change).

| Option | Effect |
|---|---|
| `--observe-fraction 0.3` | use 30% for memory (default 0.4) |
| `--observe-seconds 60` | fixed observation time instead |
| `--threshold 2.5` | alert threshold in robust spreads (default 3.0) |
| `--no-judge` | memory distances only; about one third fewer Gemma calls |
| `--explain` | one extra Gemma call per anomaly cue, adds a `Why:` line |
| `--plain-srt` | no colour tags (for players that print them literally) |
| `--render` | also write `<video>_captioned.mp4` with the text burned in (OpenCV, no audio) |

Outputs: `outputs/live/<video>_<UTC time>/`
- `<video>.srt`: subtitle track
- `report.json`: every window's caption, component scores, nearest memory caption, calibration
- `summary.txt`: anomaly intervals
- `<video>_captioned.mp4`: with `--render`

## Watch

```bash
vlc  video.mp4 --sub-file outputs/live/<run>/video.srt
mpv  video.mp4 --sub-file=outputs/live/<run>/video.srt
```

Or copy the `.srt` next to the video with the same base name (`video.mp4` + `video.srt`).
VLC, mpv and most players then load it automatically. On Windows/macOS, drag the `.srt`
onto the playing VLC window.

## Cost and limits

- Gemma calls ≈ 1 frame caption per second + 1 summary per window + 1 judge rating per
  window. A 2-minute video is ~350 calls, several minutes on one H100. An 80-minute video
  is hours; trim it or use `--no-judge`.
- The method assumes **the first 40% is normal**. An anomaly inside the observation period
  becomes "normal" memory.
- Anything new after 40% counts as a deviation, e.g. the first car in an empty street.
  The judge is there to down-weight harmless novelty, but it is not perfect.
- A subtitle segment averages the windows covering it. Labels therefore use up to 4 s of
  look-ahead: near-live, not frame-instant.
- The threshold 3.0 and equal weights are defaults. They were not tuned on labelled
  single-video data.
