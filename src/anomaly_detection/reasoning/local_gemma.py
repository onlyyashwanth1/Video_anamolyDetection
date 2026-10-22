"""
Gemma image captions and text generation, using Transformers or local Ollama.
"""

from io import BytesIO

from PIL import Image

MODEL_PATH = "/home/nverma/models/gemma-4-31B-it"

_processor = None
_model = None
_settings = {}
_client = None


def configure(config, device=None):
    """Choose a backend in each fresh worker, after its GPU has been assigned."""
    global _settings, _client
    _settings = dict(config.get("reasoning", {}))
    _client = None
    backend = _settings.get("backend", "transformers")
    if backend not in ("transformers", "ollama"):
        raise ValueError(f"Unsupported reasoning backend: {backend}")
    if backend == "ollama":
        from ollama import Client
        hosts = config.get("_ollama_hosts", {})
        host = hosts.get(device) if device is not None else _settings.get("host")
        if not host:
            raise ValueError("No Ollama endpoint configured for this worker")
        _client = Client(host=host, timeout=_settings.get("timeout", 600), trust_env=False)


def _options(max_new_tokens):
    return {"num_predict": max_new_tokens, "num_ctx": _settings.get("num_ctx", 4096),
            "num_thread": _settings.get("num_thread", 4),
            "temperature": _settings.get("temperature", 0), "seed": _settings.get("seed", 0)}


def ensure_ready():
    """Fail early on wrong model identity, missing vision, or CPU offload."""
    if _settings.get("backend", "transformers") != "ollama":
        _, model = _load()
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Gemma did not fit on one GPU; CPU/disk offload is not supported here")
        return
    model = _settings["model"]
    details = _client.show(model).model_dump()
    if "vision" not in details.get("capabilities", []):
        raise RuntimeError(f"Ollama model {model} does not support image captions")
    _client.generate(model=model, prompt="", stream=False, think=False,
                     keep_alive=-1, options=_options(1))
    loaded = _client.ps().model_dump().get("models", [])
    match = next((entry for entry in loaded if entry.get("digest") == _settings["model_digest"]), None)
    if not match:
        raise RuntimeError("Ollama loaded a different model digest; refusing incompatible checkpoints")
    if match.get("size", 0) <= 0 or match.get("size_vram", 0) < match["size"]:
        raise RuntimeError("Ollama model is not fully resident on GPU; refusing CPU offload")


def _chat(prompt, max_new_tokens, image=None):
    if _client is None:
        raise RuntimeError("Configure an Ollama endpoint before generating captions")
    message = {"role": "user", "content": prompt}
    if image is not None:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        message["images"] = [buffer.getvalue()]
    response = _client.chat(model=_settings["model"], messages=[message], stream=False,
                            think=False, keep_alive=-1, options=_options(max_new_tokens))
    text = (response.message.content or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty caption/summary")
    return text


def _load():
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

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
    if _settings.get("backend", "transformers") == "ollama":
        return _chat(prompt, max_new_tokens)
    import torch

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
    if _settings.get("backend", "transformers") == "ollama":
        return _chat(prompt, max_new_tokens, pil_image)
    import torch

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
