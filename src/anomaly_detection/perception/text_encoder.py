"""Frozen sentence encoder for Gemma captions (memory architecture, B and B+J).

MPNet (sentence-transformers/all-mpnet-base-v2) represented captions better
than CLIP's text tower in the 2026-11-08 memory-only search. Captions are
embedded as-is: mean-pooled last hidden state, L2-normalized. On CUDA the model
runs in float16, as in that search; on CPU it runs in float32.
There is no silent fallback: a missing model is an error.
"""

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class SentenceEncoder:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, batch_size=512, max_length=256,
                 local_files_only=True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only,
                                               torch_dtype=dtype).to(self.device).eval()
        self.batch_size, self.max_length = int(batch_size), int(max_length)
        self._cache = {}

    def encode(self, captions):
        """Embed captions (rows align with the input order); repeated captions are cached."""
        captions = list(captions)
        missing = [c for c in dict.fromkeys(captions) if c not in self._cache]
        if missing:
            for caption, vector in zip(missing, self._encode(missing)):
                self._cache[caption] = vector
        if not captions:
            raise ValueError("No captions to encode")
        return np.stack([self._cache[c] for c in captions])

    def _encode(self, texts):
        import torch

        order = np.argsort([len(t) for t in texts])  # similar lengths per batch
        out = None
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                rows = order[start:start + self.batch_size]
                batch = self.tokenizer([texts[r] for r in rows], padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1)
                pooled = pooled.cpu().numpy()
                if out is None:
                    out = np.empty((len(texts), pooled.shape[1]), np.float32)
                out[rows] = pooled
        return out


class PrecomputedSentenceEncoder:
    """Serve saved caption embeddings (texts, matrix), e.g. to rescore cached features."""

    def __init__(self, texts, matrix, name="precomputed"):
        self.name = name
        self.lookup = {t: i for i, t in enumerate(texts)}
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def encode(self, captions):
        missing = [c for c in captions if c not in self.lookup]
        if missing:
            raise KeyError(f"{len(missing)} captions have no saved embedding, e.g. {missing[0]!r}")
        return self.matrix[[self.lookup[c] for c in captions]]
