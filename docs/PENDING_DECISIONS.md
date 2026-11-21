# Decisions

Results table: `outputs/results/architectures/results.md`
(refresh with `.venv/bin/python scripts/collect_results.py`).

## Decided

1. **Object masking removed.** The code and the report contain no masking; every number
   was re-measured on unmasked features (`outputs/evaluations/`,
   `outputs/results/thesis_analyses_20261119/`).
2. **B + J is the proposed model.** Chapter 4 presents the 11-stage memory + judge model;
   the action + memory model and the shared configuration are the baselines (Sec. 4.8).
3. **The action + memory baseline uses 80/20**, not 95/5 (best balance of the weight sweep).
   `config/architectures/action_default.yaml` is set to `memory_weight: 0.2`; the report's
   numbers re-score the saved 95/5 runs exactly
   (`outputs/results/thesis_analyses_20261119/action_memory80_analyses.py`).
4. **Report finished:** `BTP_THESIS_VAD/main.pdf`, 75 pages. Backups of the final report and
   its results: `backup/BTP_THESIS_VAD_final_20261121.zip`, `backup/thesis_results_20261121.tgz`.
   The per-pass report backups were deleted at the end of the project.

## Still open

- [ ] **Git:** the `restructure` branch is local only. Push it to origin and/or merge it into `main`?
- [ ] **Off-server copy:** download the two files in `backup/`; `BTP_THESIS_VAD/` and `outputs/`
      are not in git.
- [ ] **First walkthrough site** (https://claude.ai/artifact/9MNzLJzMo59vms754oeoig) still shows
      "② Object-centric masking" as a step. Remove it, or leave the site as a record of the
      earlier design?
- [ ] **"Memory + Judge" site** (https://claude.ai/artifact/YUqrMTZtJ3iU8gGB5jaQW3) is private.
      Share it from the page's Share menu if wanted. Sources: `docs/site/memory_judge_src/`
      (rebuild: `python docs/site/memory_judge_src/build.py`).
- [ ] **Live 40% memory branch** (`../Video_anamolyDetection_live`, branch `live-memory-40`) has
      uncommitted edits in 8 files. Commit, discard or keep as is?
