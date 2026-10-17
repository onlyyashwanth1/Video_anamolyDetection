import torch
import torch.nn as nn
import numpy as np

class TemporalTransformer:
    """Refine a sequence of frame embeddings using a Transformer encoder.

    The transformer operates on a batch of embeddings (seq_len, d_model) and
    returns a single aggregated embedding for the event. This lightweight
    module can be plugged into the pipeline without changing the overall API.
    """

    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        self.proj = nn.Linear(d_model, d_model).to(self.device)  # identity projection, kept for flexibility

    def transform(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Encode a list of frame embeddings and return a single vector.

        Args:
            embeddings: List of ``np.ndarray`` with shape ``(d_model,)``.
        Returns:
            ``np.ndarray`` – the transformed and L2‑normalized embedding.
        """
        if not embeddings:
            raise ValueError("No embeddings provided to TemporalTransformer")
        # Stack and convert to torch tensor (seq_len, d_model)
        x = torch.stack([torch.from_numpy(e) for e in embeddings]).float().to(self.device)
        # Add a dummy batch dimension (1, seq_len, d_model)
        x = x.unsqueeze(1)
        # Apply transformer encoder
        encoded = self.encoder(x)
        # Remove batch dim and pool (mean over sequence)
        pooled = encoded.mean(dim=0).squeeze(0)
        # Optional linear projection
        out = self.proj(pooled)
        # Normalize
        out_np = out.detach().cpu().numpy()
        norm = np.linalg.norm(out_np)
        if norm > 0:
            out_np = out_np / norm
        return out_np
