"""YAML configuration with layered files.

A config may list `extends:` (paths relative to that file). Those files are
loaded first, in order, and this file's values are merged over them: nested
mappings merge key by key; lists and scalars replace. Example:

    # config/architectures/b_plus_j.yaml
    extends: [../base.yaml]
    architecture: memory_novelty

load_config(path, overlays) merges further files on top, e.g. a dataset file.
"""

from pathlib import Path

import yaml


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load(path, seen):
    path = Path(path).expanduser().resolve()
    if path in seen:
        raise ValueError(f"Circular config extends: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parents = data.pop("extends", []) or []
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    for parent in parents:
        merged = deep_merge(merged, _load(path.parent / parent, seen | {path}))
    return deep_merge(merged, data)


def load_config(path, overlays=()):
    config = _load(path, frozenset())
    for overlay in overlays:
        config = deep_merge(config, _load(overlay, frozenset()))
    return config
