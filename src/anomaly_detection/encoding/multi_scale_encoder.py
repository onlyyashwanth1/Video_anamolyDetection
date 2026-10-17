import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

from anomaly_detection.encoding.clip_encoder import ClipEncoder

class MultiScaleClipEncoder:
    """Encode an image at multiple scales and fuse the features.

    This simple implementation runs the underlying :class:`ClipEncoder` on the
    original image and on a down‑scaled version (default 0.5×). The two
    embeddings are averaged and L2‑normalized to produce a single feature
    vector. The approach is lightweight and does not require changes to the
    pretrained CLIP model.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None, scale: float = 0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = ClipEncoder(model_name=model_name, device=self.device)
        self.scale = scale
        self.resizer = Resize((int(224 * scale), int(224 * scale)))  # CLIP expects 224x224

    def _encode(self, image: np.ndarray) -> np.ndarray:
        """Encode a single image using the base CLIP encoder."""
        return self.base_encoder.encode_image(image)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode the image at original and down‑scaled resolutions.

        Args:
            image: RGB ``numpy`` array of shape ``(H, W, 3)``.

        Returns:
            Normalized fused embedding as a ``numpy`` vector.
        """
        # Original embedding
        orig_emb = self._encode(image)
        # Down‑scaled embedding
        pil_img = Image.fromarray(image)
        small_img = self.resizer(pil_img)
        small_arr = np.array(small_img)
        small_emb = self._encode(small_arr)
        # Preserve CLIP's image/text space (concatenation doubles only image dimensions).
        fused = (orig_emb + small_emb) / 2.0
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        return fused

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using the underlying ClipEncoder.

        Args:
            text: The caption or description string.

        Returns:
            Normalized text embedding as a numpy vector.
        """
        return self.base_encoder.encode_text(text)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return ClipEncoder.cosine_similarity(a, b)

    def encode_frame(self, frame):
        """Convenience wrapper for :class:`anomaly_detection.utils.types.Frame`."""
        from anomaly_detection.utils.types import EncodedFrame
        return EncodedFrame(frame=frame, embedding=self.encode_image(frame.image))
