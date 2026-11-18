"""Observe the first part of one video, then score the rest against it."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from anomaly_detection.live.self_memory import SelfMemory, boundary_frame, split_windows, time_guard
from anomaly_detection.live.subtitles import ANOMALY, build_cues, build_segments, write_srt


def _caption(processor, window, frames, cache):
    from anomaly_detection.familiarisation.domain_familiarisation import caption_windows
    caption_windows([window], processor.encoder, frames, processor.config.get("captioning", {}),
                    strict=True, cache=cache)
    return window.description


def _stamp(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def run_live(video_path, config, output_dir, log=print, render_video=False):
    from anomaly_detection.live.text_encoder import CaptionEncoder
    from anomaly_detection.processing import VideoEventProcessor
    from anomaly_detection.reasoning.ollama_runtime import caption_session

    live = config["live"]
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    log(f"Reading and encoding {video_path.name} (CLIP, {config['video']['target_fps']} frame/s)...")
    processor = VideoEventProcessor(config)
    video = processor.process(str(video_path), caption=False)
    fps, duration = video.fps, video.frame_count / video.fps
    boundary = boundary_frame(video.frame_count, fps, live.get("observe_fraction"), live.get("observe_seconds"))
    observe_until = boundary / fps
    memory, test = split_windows(video.events, boundary)
    log(f"Video {duration:.1f} s. Observation (memory) 0-{observe_until:.1f} s: {len(memory)} windows. "
        f"Testing {test[0].start_time:.1f}-{duration:.1f} s: {len(test)} windows.")
    text_encoder = CaptionEncoder(live.get("text_encoder"), processor.encoder)

    judge_on = bool(live.get("judge", {}).get("enabled", False)) and float(live.get("judge_weight", 0)) > 0
    records = {"memory": [], "test": []}
    with caption_session(config):
        cache = {}
        log("\n--- Phase 1: observing (no alerts; building normal memory) ---")
        for i, window in enumerate(memory, 1):
            caption = _caption(processor, window, video.frames, cache)
            log(f"[{_stamp(window.start_time)}] OBSERVE {i}/{len(memory)}: {caption}")
        memory_text = text_encoder.encode([w.description for w in memory])
        memory_visual = np.stack([w.average_embedding for w in memory])

        judge, memory_judge = None, None
        if judge_on:
            from anomaly_detection.live.judge import GemmaJudge, retrieve
            judge = GemmaJudge(config, live["judge"].get("memories", 8))
            guard = time_guard(memory, int(round(float(live.get("calibration_guard_seconds", 4.0)) * fps)))
            retrieved = retrieve(memory_text, memory_text, [w.description for w in memory],
                                 judge.memories, guard)
            log("Rating observation windows with the memory-aware judge (for calibration)...")
            memory_judge = [judge.rate(w.description, r) for w, r in zip(memory, retrieved)]

        model = SelfMemory.build(memory, memory_visual, memory_text, live, fps, memory_judge)
        threshold = float(live.get("threshold", 3.0))
        log(f"Memory frozen: {len(memory)} windows, {len(set(model.captions))} distinct captions. "
            f"Alert when deviation > {threshold:.1f} (observation max {model.calibration['observation_max_z']:.1f}).")
        for i, window in enumerate(memory):
            records["memory"].append({"start_time": window.start_time, "end_time": window.end_time,
                                      "caption": window.description,
                                      "judge": None if memory_judge is None else float(memory_judge[i])})

        log("\n--- Phase 2: testing against the frozen memory ---")
        deviations = []
        for window in test:
            caption = _caption(processor, window, video.frames, cache)
            text = text_encoder.encode([caption])
            rating = None
            if judge is not None:
                rating = judge.rate(caption, retrieve(text, memory_text, model.captions, judge.memories)[0])
            scored = model.score(window.average_embedding[None], text,
                                 None if rating is None else np.asarray([rating]))
            deviation = float(scored["deviation"][0])
            deviations.append(deviation)
            flag = "ANOMALY" if deviation > threshold else "normal "
            log(f"[{_stamp(window.start_time)}] {flag} deviation {deviation:5.2f} | {caption}")
            records["test"].append({
                "start_time": window.start_time, "end_time": window.end_time, "caption": caption,
                "deviation": deviation, "window_anomaly": deviation > threshold, "judge": rating,
                "nearest_memory_caption": scored["nearest_memory_caption"][0],
                **{key: float(value[0]) for key, value in scored.items()
                   if key not in ("deviation", "nearest_memory_caption")},
            })

        captions = {("memory", i): w.description for i, w in enumerate(memory)}
        captions.update({("test", j): w.description for j, w in enumerate(test)})
        segments = build_segments(memory, test, deviations, duration)
        cues = build_cues(segments, captions, threshold, float(live.get("srt", {}).get("cue_seconds", 4.0)))

        if live.get("explain_anomalies", False):
            from anomaly_detection.reasoning.llm_reasoning import explain_anomaly
            from anomaly_detection.utils.types import AnomalyResult
            notebook = list(dict.fromkeys(model.captions))[:40]
            for n, cue in enumerate(cues):
                if cue.label != ANOMALY:
                    continue
                context = " ".join(c.caption for c in cues[max(0, n - 2):n] if c.caption)
                result = explain_anomaly(AnomalyResult(test[cue.window_index], cue.deviation, True, None),
                                         notebook, context=context)
                cue.explanation = result.explanation
                cue.extra["gemma_verdict"] = "ANOMALY" if result.is_anomaly else "NORMAL"

    stem = video_path.stem
    srt_path = output_dir / f"{stem}.srt"
    colors = bool(live.get("srt", {}).get("colors", True))
    write_srt(cues, srt_path, threshold, observe_until, colors)
    anomalies = [{"start_time": c.start, "end_time": c.end, "max_deviation": c.deviation,
                  "caption": c.caption, "explanation": c.explanation, **c.extra}
                 for c in cues if c.label == ANOMALY]
    merged = []
    for item in anomalies:
        if merged and abs(merged[-1]["end_time"] - item["start_time"]) < 1e-6:
            merged[-1]["end_time"] = item["end_time"]
            if item["max_deviation"] > merged[-1]["max_deviation"]:
                # Describe the interval by its most anomalous moment.
                merged[-1].update(max_deviation=item["max_deviation"], caption=item["caption"])
        else:
            merged.append({k: item[k] for k in ("start_time", "end_time", "max_deviation", "caption")})
    report = {
        "video": str(video_path), "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 1),
        "fps": fps, "frame_count": video.frame_count, "duration_seconds": duration,
        "observe_until_seconds": observe_until, "boundary_frame": boundary,
        "memory_windows": len(memory), "test_windows": len(test),
        "skipped_boundary_windows": len(video.events) - len(memory) - len(test),
        "text_encoder": text_encoder.name, "masking": processor.masker.enabled,
        "judge": judge_on, "threshold": threshold, "weights": model.weights, "fusion": model.fusion,
        "calibration": model.calibration, "live_settings": live,
        "anomaly_intervals": merged, "anomaly_cues": anomalies,
        "memory": records["memory"], "test": records["test"],
        "cues": [{"start": c.start, "end": c.end, "label": c.label, "caption": c.caption,
                  "deviation": c.deviation, "explanation": c.explanation} for c in cues],
        "notes": [
            "Memory is built only from windows ending before the boundary; no test window enters memory.",
            "Deviation is in units of the observation period's robust spread (median / IQR), held out by a time guard.",
            "A subtitle segment's deviation averages the overlapping windows covering it (up to 4 s look-ahead).",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = [f"Video: {video_path}", f"Duration {duration:.1f} s; memory 0-{observe_until:.1f} s "
               f"({len(memory)} windows); tested {len(test)} windows", f"Text encoder: {text_encoder.name}; "
               f"judge: {judge_on}; threshold {threshold}", "Anomaly intervals:"]
    summary += [f"  {_stamp(a['start_time'])}-{_stamp(a['end_time'])}  max deviation {a['max_deviation']:.1f}  "
                f"{a['caption']}" for a in merged] or ["  none"]
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    log("\n" + "\n".join(summary))
    log(f"\nSubtitles: {srt_path}")
    if render_video:
        from anomaly_detection.live.render import render
        out = render(video_path, cues, output_dir / f"{stem}_captioned.mp4", threshold, observe_until)
        log(f"Captioned video (no audio): {out}")
    return report, srt_path
