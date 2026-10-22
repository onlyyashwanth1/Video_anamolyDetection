"""Ollama routing, GPU residency, provenance, and owned-server cleanup."""

import copy
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image
import pytest

from anomaly_detection.reasoning import local_gemma as gemma
from anomaly_detection.reasoning import ollama_runtime as runtime
from anomaly_detection.evaluation.bank import model_spec


def response(value):
    return SimpleNamespace(model_dump=lambda: value)


@pytest.fixture
def configured(monkeypatch):
    import ollama
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(ollama, "Client", factory)
    config = {"reasoning": {"backend": "ollama", "model": "gemma4:31b-it-q8_0",
                            "model_digest": "abc", "quantization": "Q8_0"},
              "_ollama_hosts": {"3": "http://127.0.0.1:11501"}}
    gemma.configure(config, "3")
    yield client, factory, config
    gemma.configure({})


def test_routes_images_and_text_without_transformers(configured):
    client, factory, _ = configured
    factory.assert_called_once_with(host="http://127.0.0.1:11501", timeout=600, trust_env=False)
    client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=" people walking "))
    assert gemma.generate_caption(Image.new("RGB", (3, 2), "red"), "Describe", 60) == "people walking"
    call = client.chat.call_args.kwargs
    assert call["think"] is False and call["stream"] is False
    assert call["model"] == "gemma4:31b-it-q8_0"
    assert call["options"]["num_predict"] == 60
    image = Image.open(BytesIO(call["messages"][0]["images"][0]))
    assert image.size == (3, 2) and image.getpixel((0, 0)) == (255, 0, 0)
    assert gemma.generate_text("Summarize", 40) == "people walking"
    assert "images" not in client.chat.call_args.kwargs["messages"][0]
    client.chat.return_value.message.content = ""
    with pytest.raises(RuntimeError, match="empty"):
        gemma.generate_text("Summarize")


def test_readiness_requires_vision_matching_digest_and_full_gpu(configured):
    client, _, _ = configured
    client.show.return_value = response({"capabilities": ["completion", "vision"]})
    entry = {"digest": "abc", "size": 100, "size_vram": 100}
    client.ps.return_value = response({"models": [entry]})
    gemma.ensure_ready()
    entry["size_vram"] = 50
    with pytest.raises(RuntimeError, match="CPU offload"):
        gemma.ensure_ready()
    entry["digest"] = "changed"
    with pytest.raises(RuntimeError, match="different model digest"):
        gemma.ensure_ready()
    client.show.return_value = response({"capabilities": ["completion"]})
    with pytest.raises(RuntimeError, match="image captions"):
        gemma.ensure_ready()


def test_missing_worker_endpoint_is_rejected():
    with pytest.raises(ValueError, match="endpoint"):
        gemma.configure({"reasoning": {"backend": "ollama"}}, "0")
    gemma.configure({})


def test_model_resolution_and_checkpoint_identity(tmp_path):
    models = tmp_path / "models"
    (models / "blobs").mkdir(parents=True)
    manifest = models / "manifests/registry.ollama.ai/library/gemma4/q8"
    manifest.parent.mkdir(parents=True)
    digest = "sha256:" + "a" * 64
    raw = json.dumps({"file_type": "Q8_0"}).encode()
    (models / "blobs" / digest.replace(":", "-")).write_bytes(raw)
    manifest.write_text(json.dumps({"config": {"digest": digest, "size": len(raw)}, "layers": []}))
    config = {"reasoning": {"backend": "ollama", "model": "gemma4:q8", "models_dir": str(models)},
              "encoder": {}, "video": {"target_fps": 1}, "segmentation": {},
              "familiarisation": {"min_caption_occurrences": 1}}
    runtime.resolve_model(config)
    before = model_spec(config)
    other = copy.deepcopy(config)
    other["_ollama_hosts"] = {"7": "http://127.0.0.1:12000"}
    assert model_spec(other) == before
    other["reasoning"]["model_digest"] = "changed"
    assert model_spec(other) != before
    other = copy.deepcopy(config)
    other["reasoning"]["num_ctx"] = 8192
    assert model_spec(other) != before
    (models / "blobs" / digest.replace(":", "-")).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="incomplete"):
        runtime.resolve_model(config)


@pytest.mark.parametrize("fail_second", [False, True])
def test_managed_servers_pin_devices_and_cleanup(tmp_path, monkeypatch, fail_second):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/ollama")
    monkeypatch.setattr(runtime.socket, "socket", MagicMock())
    opener = MagicMock()
    opener.open.side_effect = lambda *a, **k: BytesIO(b'{"version":"test"}')
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a: opener)
    processes = [MagicMock(pid=101), MagicMock(pid=102)]
    processes[0].poll.return_value = None
    processes[1].poll.return_value = 1 if fail_second else None
    spawn = MagicMock(side_effect=processes)
    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    stop = MagicMock()
    monkeypatch.setattr(runtime, "_stop", stop)
    config = {"reasoning": {"backend": "ollama", "models_dir": str(tmp_path)}}
    if fail_second:
        with pytest.raises(RuntimeError, match="exited"):
            with runtime.ollama_servers(["GPU-a", "GPU-b"], config, tmp_path / "logs"):
                pytest.fail("Should not yield after failed startup")
    else:
        with runtime.ollama_servers(["GPU-a", "GPU-b"], config, tmp_path / "logs"):
            assert config["_ollama_hosts"] == {"GPU-a": "http://127.0.0.1:11500",
                                                "GPU-b": "http://127.0.0.1:11501"}
    assert "_ollama_hosts" not in config
    assert stop.call_count == 2
    first, second = spawn.call_args_list
    assert first.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a"
    assert second.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert first.kwargs["env"]["OLLAMA_NUM_PARALLEL"] == "1"
    assert first.kwargs["start_new_session"] is True
