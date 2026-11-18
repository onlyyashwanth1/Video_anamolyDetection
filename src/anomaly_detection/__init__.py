"""Training-free video anomaly detection with frozen foundation models.

Layout:
    perception/   video -> 1 fps frames -> CLIP window embeddings -> Gemma captions
                  (+ MPNet caption embeddings for the memory architecture)
    memory/       normal memory: per-video observation memory and the saved bank
    scoring/      the three architectures: observe_detect, action, memory_novelty (B, B+J)
    temporal/     window scores -> per-frame scores (pooling, smoothing)
    reasoning/    Gemma via Ollama or Transformers; optional explanations
    evaluation/   datasets, video cache, multi-GPU runner, metrics
    pipeline.py   single-video entry point
    config.py     layered YAML configs (config/base.yaml + architecture + dataset)

See docs/ARCHITECTURES.md.
"""
