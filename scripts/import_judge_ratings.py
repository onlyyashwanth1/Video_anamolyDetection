#!/usr/bin/env python3
"""Merge saved judge ratings (prompt -> expected digit JSON) into the judge cache.

The B+J study rated prompts with the same model, prompt text, num_ctx 2048,
temperature 0 and seed 0 (outputs/results/memory_only_search_20261107/llm_rag_*.json),
so those ratings are valid cache entries. Existing cache entries are kept.

    python scripts/import_judge_ratings.py outputs/results/memory_only_search_20261107/llm_rag_*.json
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from anomaly_detection.config import load_config
    from anomaly_detection.scoring import judge_cache_path
    from anomaly_detection.scoring.judge import PROMPT, RatingCache

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--architecture", default="b_plus_j")
    args = parser.parse_args(argv)
    config = load_config(ROOT / "config/architectures" / f"{args.architecture}.yaml")
    cache = RatingCache(judge_cache_path(config))
    prefix = PROMPT.split("{memories}")[0]
    before = len(cache.values)
    for path in args.files:
        ratings = json.loads(path.read_text())
        valid = {p: float(v) for p, v in ratings.items() if p.startswith(prefix) and v == v}
        conflicts = sum(1 for p, v in valid.items() if p in cache.values and abs(cache.values[p] - v) > 1e-9)
        for prompt, value in valid.items():
            cache.values.setdefault(prompt, value)
        print(f"{path.name}: {len(valid)} ratings ({len(ratings) - len(valid)} skipped, {conflicts} conflicting kept as cached)")
    cache.save()
    print(f"{cache.path}: {before} -> {len(cache.values)} prompts")


if __name__ == "__main__":
    main()
