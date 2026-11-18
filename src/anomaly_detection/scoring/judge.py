"""Memory-aware Gemma judge: the J in B+J.

For a window caption, retrieve the `memories` most similar distinct captions
from normal memory (MPNet cosine) and ask Gemma, with a fixed generic prompt that
names no dataset or category, how abnormal the new moment is from 0 to 9. The
rating is the expected digit under Gemma's next-token probabilities.

Calibration: rate `calibration_windows` randomly chosen normal-memory windows
(seeded), each retrieving only from captions that also occur in OTHER memory
videos, and z-score test ratings with that normal mean and standard deviation
(most normal ratings are ~0, so an IQR scale would collapse).

Ratings are cached by exact prompt text in a JSON file, so re-runs and other
datasets reuse every rating already made.
"""

import concurrent.futures
import json
import logging
import math
import os
from pathlib import Path
import time
import urllib.request

import numpy as np

LOGGER = logging.getLogger(__name__)
PROMPT = ("You are reviewing moments described from a video.\n"
          "Typical normal moments from this setting:\n{memories}\n"
          "New moment: {caption}\n"
          "Compared with the typical normal moments, how likely is it that something abnormal is "
          "happening in the new moment, such as an activity that is unusual or out of place for this "
          "setting, or something harmful or dangerous? Reply with one digit from 0 (clearly normal) "
          "to 9 (clearly abnormal).")
DIGITS = [str(d) for d in range(10)]


def format_prompt(caption, memories):
    return PROMPT.format(memories="\n".join(f"- {m}" for m in memories), caption=caption)


def expected_digit(response):
    """Expected rating from an Ollama /api/chat response with top_logprobs."""
    probabilities = {}
    for candidate in ((response.get("logprobs") or [{}])[0].get("top_logprobs") or []):
        token = candidate.get("token", "").strip()
        if token in DIGITS:
            probabilities[token] = probabilities.get(token, 0.0) + math.exp(candidate["logprob"])
    mass = sum(probabilities.values())
    if mass > 0:
        return sum(int(t) * p for t, p in probabilities.items()) / mass
    text = (response.get("message", {}).get("content") or "").strip()
    if text[:1] in DIGITS:
        return float(text[0])
    raise RuntimeError(f"Judge did not return a digit: {text!r}")


def retrieve(query_vectors, memory_vectors, memory_texts, k, query_groups=None, entry_only_group=None,
             candidates=None, chunk=1024, device="cpu"):
    """Up to k distinct memory texts most similar to each query (unit vectors).

    The `candidates` (4k) most similar entries are taken in order of similarity and
    de-duplicated. With query_groups and entry_only_group, an entry is not
    retrievable by a query when it occurs only in the query's own group
    (entry_only_group[e] is that group, or -1 when it occurs in several groups).

    On a CUDA device similarities are computed in float16, exactly as in the
    B+J study, so its cached judge ratings are reused; near-ties can order
    differently in float32 on the CPU, which then needs new ratings.
    """
    candidates = candidates or 4 * k
    take = min(candidates, len(memory_vectors))
    queries = np.asarray(query_vectors, dtype=np.float32)
    if str(device) != "cpu":
        import torch
        memory_gpu = torch.as_tensor(np.asarray(memory_vectors, dtype=np.float32), device=device).half()
        only = None if query_groups is None else torch.as_tensor(np.asarray(entry_only_group), device=device)
    else:
        memory = np.asarray(memory_vectors, dtype=np.float32)
    picked_all = []
    for start in range(0, len(queries), chunk):
        if str(device) != "cpu":
            with torch.inference_mode():
                sims = torch.as_tensor(queries[start:start + chunk], device=device).half() @ memory_gpu.T
                if query_groups is not None:
                    own = torch.as_tensor(np.asarray(query_groups[start:start + chunk]), device=device)
                    sims.masked_fill_(only[None, :] == own[:, None], float("-inf"))
                values, indices = torch.topk(sims, take, dim=1)
                ordered = np.where(np.isfinite(values.float().cpu().numpy()), indices.cpu().numpy(), -1)
        else:
            sims = queries[start:start + chunk] @ memory.T
            if query_groups is not None:
                own = np.asarray(query_groups[start:start + chunk])
                sims[np.asarray(entry_only_group)[None, :] == own[:, None]] = -np.inf
            top = np.argpartition(-sims, take - 1, axis=1)[:, :take]
            ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, 1), 1, kind="stable"), 1)
            ordered = np.where(np.isfinite(np.take_along_axis(sims, ordered, 1)), ordered, -1)
        for row in ordered:
            picked = []
            for j in row:
                if j < 0:
                    break
                if memory_texts[j] not in picked:
                    picked.append(memory_texts[j])
                if len(picked) == k:
                    break
            picked_all.append(picked)
    return picked_all


class RatingCache:
    """prompt -> expected rating, persisted as JSON (atomic replace)."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.values = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.values))
        os.replace(temporary, self.path)


class OllamaRater:
    """Single-token ratings from the Ollama servers started by ollama_runtime."""

    def __init__(self, config, num_ctx=2048):
        hosts = config.get("_ollama_hosts") or {}
        if not hosts:
            raise RuntimeError("The judge needs running Ollama servers (ollama_servers / caption_session)")
        self.hosts = list(hosts.values())
        settings = config["reasoning"]
        self.model, self.timeout = settings["model"], settings.get("timeout", 600)
        self.options = dict(num_predict=1, temperature=0, seed=settings.get("seed", 0),
                            num_ctx=int(num_ctx), num_thread=settings.get("num_thread", 4))

    def rate_one(self, host, prompt):
        payload = dict(model=self.model, messages=[{"role": "user", "content": prompt}], stream=False,
                       think=False, keep_alive=-1, logprobs=True, top_logprobs=20, options=self.options)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    return expected_digit(json.load(response))
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    def rate(self, prompts, on_result=None):
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.hosts)) as pool:
            futures = {pool.submit(self.rate_one, self.hosts[i % len(self.hosts)], p): p
                       for i, p in enumerate(prompts)}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
                if on_result is not None:
                    on_result(futures[future], results[futures[future]])
        return results


class MemoryJudge:
    def __init__(self, settings, memory, text_encoder, rater=None, cache=None):
        self.memory, self.text_encoder, self.rater = memory, text_encoder, rater
        self.k = int(settings.get("memories", 8))
        self.calibration_windows = int(settings.get("calibration_windows", 2000))
        self.seed = int(settings.get("seed", 0))
        self.cache = cache if cache is not None else RatingCache(None)
        self.calibration = None

    def ratings(self, prompts):
        missing = [p for p in dict.fromkeys(prompts) if p not in self.cache.values]
        if missing:
            if self.rater is None:
                raise RuntimeError(f"{len(missing)} judge prompts are not cached and no Gemma rater is set")
            LOGGER.info("Rating %d new judge prompts with Gemma", len(missing))
            done = [0]

            def store(prompt, value):
                self.cache.values[prompt] = value
                done[0] += 1
                if done[0] % 500 == 0:
                    self.cache.save()
            self.rater.rate(missing, on_result=store)
            self.cache.save()
        return np.asarray([self.cache.values[p] for p in prompts], dtype=np.float64)

    def calibration_prompts(self):
        """Prompts for the seeded sample of normal-memory windows (own video excluded)."""
        memory = self.memory
        count = min(self.calibration_windows, len(memory.window_entry))
        rows = np.sort(np.random.default_rng(self.seed).choice(len(memory.window_entry), count, replace=False))
        groups_per_entry = {}
        for entry, group in zip(memory.window_entry.tolist(), memory.groups.tolist()):
            groups_per_entry.setdefault(entry, set()).add(group)
        only_group = np.full(len(memory.entry_texts), -1)
        for entry, groups in groups_per_entry.items():
            if len(groups) == 1:
                only_group[entry] = next(iter(groups))
        entries = memory.window_entry[rows]
        captions = [memory.entry_texts[e] for e in entries]
        retrieved = retrieve(memory.entry_text_vectors[entries], memory.entry_text_vectors,
                             memory.entry_texts, self.k, query_groups=memory.groups[rows],
                             entry_only_group=only_group, device=memory.device)
        return [format_prompt(c, m) for c, m in zip(captions, retrieved)]

    def calibrate(self):
        if self.calibration is None:
            ratings = self.ratings(self.calibration_prompts())
            self.calibration = {"mean": float(ratings.mean()), "std": float(max(ratings.std(), 1e-3)),
                                "windows": len(ratings)}
            LOGGER.info("Judge calibration on %d normal-memory windows: mean %.4f, std %.4f",
                        len(ratings), self.calibration["mean"], self.calibration["std"])
        return self.calibration

    def prompts(self, captions):
        unique = list(dict.fromkeys(captions))
        retrieved = retrieve(self.text_encoder.encode(unique), self.memory.entry_text_vectors,
                             self.memory.entry_texts, self.k, device=self.memory.device)
        by_caption = {c: format_prompt(c, m) for c, m in zip(unique, retrieved)}
        return [by_caption[c] for c in captions]

    def score(self, captions):
        """(raw expected ratings, calibrated z) for each caption."""
        calibration = self.calibrate()
        raw = self.ratings(self.prompts(captions))
        return raw, (raw - calibration["mean"]) / calibration["std"]
