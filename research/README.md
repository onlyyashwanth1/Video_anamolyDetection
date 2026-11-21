# Research scripts (frozen)

These scripts produced the searches, ablations and analyses behind the thesis
results. They were written against the package layout before the restructure
(`anomaly_detection.inference`, `.encoding`, `.evaluation.bank`, ...), so they
run from the snapshot commit `a3cb86e`, not from this tree:

```bash
git worktree add ../vad_snapshot a3cb86e    # one already exists at ../vad_a3cb86e
```

On the current tree `pytest research/tests` skips these tests with a message
(`research/tests/conftest.py`); run them from the snapshot instead.

| Script | Purpose |
|---|---|
| `scripts/search_*_configs.py`, `search_shared_prompts.py`, `search_temporal_designs.py` | offline configuration, prompt and temporal searches behind `shared_best` |
| `scripts/prepare_shared_prompt_features.py` | cached window features used by those searches |
| `scripts/evaluate_shared_model.py`, `evaluate_shared_external.py`, `compare_*` | shared-model evaluation and comparisons |
| `scripts/analyze_*`, `audit_xd_misses.py` | ablation and error analyses |
| `scripts/extract_unmasked_clip.py` | the unmasked CLIP features used by models B/B+J (its old `masked` mode needs the deleted YOLO-World weights `yolov8s-worldv2.pt`) |
| `scripts/evaluate_ucf_crime.py`, `evaluate_xd_violence.py`, `evaluate_pipeline.py`, `run_pipeline.py` | older serial evaluators and demos, superseded by `scripts/evaluate.py` / `run_video.py` |
| `config/ucf_action_prompts_expanded.yaml` | the 26 + 26 prompts of the action + memory baseline (also in `config/architectures/action_default.yaml`) |
| `config/vocab/` | concept vocabularies and Gemma hazard ratings from the knowledge-prior study |
| `tests/test_offline_config_search.py` | tests for the offline search code |

The memory-only search that defined B and B+J lives with its results in
`outputs/results/memory_only_search_20261107/` (report.md, report_knowledge.md).
Its B and B+J scoring is now `src/anomaly_detection/scoring/memory_novelty.py` and `judge.py`.
Its scripts (and `outputs/results/memory_bank_uses_20261107/`) read the captions and video
sources from the earlier memory banks `outputs/memory/*_visual_text_*.npz`, so those banks
are kept.
