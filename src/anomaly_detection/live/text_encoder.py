"""Frozen sentence encoder for Gemma captions.

MPNet represented captions better than CLIP's text tower in the
memory-only search (2026-11-08). If it is not available locally, CLIP
text features are used instead, and the run report says so.
"""

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class CaptionEncoder:
    def __init__(self, model_name, clip_encoder, device=None):
        self.name = model_name
        self.clip = clip_encoder
        self.model = None
        if model_name and model_name != "clip":
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device).eval()
            except Exception as error:  # missing weights or offline machine
                LOGGER.warning("Sentence encoder %s unavailable (%s); using CLIP text", model_name, error)
                self.model = None
        if self.model is None:
            self.name = "clip"
        self._cache = {}

    def encode(self, captions):
        missing = [c for c in dict.fromkeys(captions) if c not in self._cache]
        if missing:
            for caption, vector in zip(missing, self._encode(missing)):
                self._cache[caption] = vector
        return np.stack([self._cache[c] for c in captions])

    def _encode(self, captions):
        if self.model is None:
            return np.stack([self.clip.encode_text(c) for c in captions])
        import torch
        outputs = []
        with torch.no_grad():
            for start in range(0, len(captions), 64):
                batch = self.tokenizer(captions[start:start + 64], padding=True, truncation=True,
                                       max_length=256, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1)
                outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs)
