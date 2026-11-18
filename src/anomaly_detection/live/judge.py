"""Memory-aware Gemma judge (the "J" in B+J).

Gemma sees the K most similar captions from this scene's normal memory
and rates a new caption from 0 (clearly normal) to 9 (clearly abnormal).
The prompt is generic and names no dataset or category. The score is the
expected digit from the token probabilities, as in
outputs/results/memory_only_search_20261107/rag_judge.py.
"""

import json
import math
import time
import urllib.request

import numpy as np

PROMPT = ("You are reviewing moments described from a video.\n"
          "Typical normal moments from this setting:\n{memories}\n"
          "New moment: {caption}\n"
          "Compared with the typical normal moments, how likely is it that something abnormal is "
          "happening in the new moment, such as an activity that is unusual or out of place for this "
          "setting, or something harmful or dangerous? Reply with one digit from 0 (clearly normal) "
          "to 9 (clearly abnormal).")
DIGITS = [str(d) for d in range(10)]


def retrieve(query_vectors, memory_vectors, memory_captions, k, excluded=None):
    """Up to k distinct memory captions most similar to each query."""
    sims = np.asarray(query_vectors) @ np.asarray(memory_vectors).T
    if excluded is not None:
        sims = np.where(excluded, -np.inf, sims)
    picked_all = []
    for row in sims:
        picked = []
        for index in np.argsort(-row):
            if not np.isfinite(row[index]) or len(picked) == k:
                break
            if memory_captions[index] not in picked:
                picked.append(memory_captions[index])
        picked_all.append(picked)
    return picked_all


class GemmaJudge:
    def __init__(self, config, memories=8):
        hosts = config.get("_ollama_hosts") or {}
        if not hosts:
            raise RuntimeError("The judge needs the Ollama server started by caption_session")
        self.host = next(iter(hosts.values()))
        settings = config["reasoning"]
        self.model, self.memories = settings["model"], int(memories)
        self.options = dict(num_predict=1, temperature=0, seed=settings.get("seed", 0),
                            num_ctx=settings.get("num_ctx", 4096), num_thread=settings.get("num_thread", 4))
        self.timeout = settings.get("timeout", 600)
        self._cache = {}

    def rate(self, caption, memories):
        prompt = PROMPT.format(memories="\n".join(f"- {m}" for m in memories) or "- (none)",
                               caption=caption)
        if prompt not in self._cache:
            self._cache[prompt] = self._rate(prompt)
        return self._cache[prompt]

    def _rate(self, prompt):
        payload = dict(model=self.model, messages=[{"role": "user", "content": prompt}], stream=False,
                       think=False, keep_alive=-1, logprobs=True, top_logprobs=20, options=self.options)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    out = json.load(response)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        probabilities = {}
        for candidate in ((out.get("logprobs") or [{}])[0].get("top_logprobs") or []):
            token = candidate.get("token", "").strip()
            if token in DIGITS:
                probabilities[token] = probabilities.get(token, 0.0) + math.exp(candidate["logprob"])
        if probabilities:
            mass = sum(probabilities.values())
            return sum(int(t) * p for t, p in probabilities.items()) / mass
        text = (out.get("message", {}).get("content") or "").strip()
        if text[:1] in DIGITS:
            return float(text[0])
        raise RuntimeError(f"Judge did not return a digit: {text!r}")
