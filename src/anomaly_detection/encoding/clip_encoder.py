"""
Step 2: Frozen Foundational Encoder (CLIP)
--------------------------------------------
Paper reference: "FROZEN FOUNDATIONAL ENCODER (Zero-Shot Feature
Extraction: e.g. CLIP)"

Wraps a pretrained CLIP model and turns images (or text) into
"fingerprints" - fixed-length numeric vectors where things with similar
meaning end up close together, regardless of whether they started as an
image or a sentence. This is what later lets the system compare "what the
camera sees" against "what normal was described as, in words."

This model is NEVER fine-tuned or updated (that's what "frozen" means).
We only ever call it in inference mode (torch.no_grad()).
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from anomaly_detection.utils.types import Frame, EncodedFrame


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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # inference mode only - frozen, no training
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        image: an RGB numpy array (H, W, 3)
        returns: a normalized 1D numpy vector (the "fingerprint")
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        raw_output = self.model.get_image_features(**inputs)
        features = _unwrap_features(raw_output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def encode_frame(self, frame: Frame) -> EncodedFrame:
        """Convenience wrapper: Frame in, EncodedFrame out."""
        return EncodedFrame(frame=frame, embedding=self.encode_image(frame.image))

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        text: a plain-English sentence, e.g. "an empty hallway"
        returns: a normalized 1D numpy vector, in the SAME space as encode_image's output
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        raw_output = self.model.get_text_features(**inputs)
        features = _unwrap_features(raw_output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Both vectors are already normalized, so this is just a dot product."""
        return float(np.dot(a, b))
