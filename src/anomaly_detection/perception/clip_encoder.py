"""Frozen CLIP encoders for images and text (never fine-tuned).

ClipEncoder embeds one image or sentence into CLIP's shared space.
MultiScaleClipEncoder, the encoder every pipeline uses, averages the image
embedding at full and half resolution; its text embeddings are plain CLIP text.
Frames are encoded unmasked: the object-masking step was removed after the
2026-11-08 ablation showed it hurt memory scoring and XD-Violence.
"""

import numpy as np
import torch
from PIL import Image

from anomaly_detection.datatypes import EncodedFrame, Frame


def _unwrap_features(output):
    """Handles both raw-tensor and wrapped-object return types across transformers versions."""
    if torch.is_tensor(output):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(output, attr):
            return getattr(output, attr)
    return output[0]


class ClipEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """RGB array (H, W, 3) -> normalized embedding."""
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        features = _unwrap_features(self.model.get_image_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def encode_frame(self, frame: Frame) -> EncodedFrame:
        return EncodedFrame(frame=frame, embedding=self.encode_image(frame.image))

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Sentence -> normalized embedding in the same space as encode_image."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        features = _unwrap_features(self.model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=256) -> np.ndarray:
        """Same text features as encode_text, batched."""
        if batch_size < 1 or not texts:
            raise ValueError("Expected texts and a positive batch size")
        outputs = []
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(text=list(texts[start:start + batch_size]), return_tensors="pt",
                                    padding=True).to(self.device)
            features = _unwrap_features(self.model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            outputs.append(features.cpu().numpy())
        return np.concatenate(outputs)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Both vectors are normalized, so this is a dot product."""
        return float(np.dot(a, b))


class MultiScaleClipEncoder:
    """Average CLIP image embeddings at full and `scale` resolution, then renormalize.

    Averaging (not concatenating) keeps image embeddings in CLIP's image/text space.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None,
                 scale: float = 0.5):
        from torchvision.transforms import Resize

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_encoder = ClipEncoder(model_name=model_name, device=self.device)
        self.scale = scale
        self.resizer = Resize((int(224 * scale), int(224 * scale)))

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        full = self.base_encoder.encode_image(image)
        small = self.base_encoder.encode_image(np.array(self.resizer(Image.fromarray(image))))
        fused = (full + small) / 2.0
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 0 else fused

    def encode_frame(self, frame: Frame) -> EncodedFrame:
        return EncodedFrame(frame=frame, embedding=self.encode_image(frame.image))

    def encode_text(self, text: str) -> np.ndarray:
        return self.base_encoder.encode_text(text)

    def encode_texts(self, texts, batch_size=256) -> np.ndarray:
        return self.base_encoder.encode_texts(texts, batch_size)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return ClipEncoder.cosine_similarity(a, b)
