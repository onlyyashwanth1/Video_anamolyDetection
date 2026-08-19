"""
Local Gemma 4 31B loader (replaces Ollama).
Loads the model once and reuses it for both text-only prompts
(reasoning) and image+text prompts (captioning).
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = "/home/nverma/models/gemma-4-31B-it"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        print("[local_gemma] Loading model (first call only)...")
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        print("[local_gemma] Model ready on", next(_model.parameters()).device)
    return _processor, _model


def generate_text(prompt: str, max_new_tokens: int = 150) -> str:
    """Text-only generation. Used for explain_anomaly()."""
    processor, model = _load()

    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def generate_caption(pil_image: Image.Image, prompt: str, max_new_tokens: int = 60) -> str:
    """Image + text generation. Used for caption_event()."""
    processor, model = _load()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()