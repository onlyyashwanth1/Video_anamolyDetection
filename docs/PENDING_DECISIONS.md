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
4. **Report finished:** `BTP_THESIS_VAD/BTP_report.pdf`, 75 pages. Backups of the final report and
   its results: `backup/BTP_THESIS_VAD_final_20261121.zip`, `backup/thesis_results_20261121.tgz`.
   The per-pass report backups were deleted at the end of the project.
5. **Git:** `restructure` was pushed to GitHub and merged into `main` (fast-forward). The report
   (`BTP_THESIS_VAD/`, including its PDFs) is tracked in git.
6. **Walkthrough sites** are shared with anyone who has the link and are listed in `README.md`.
   The first site is kept as a record of the earlier design (95 / 5 weight, masking step).
7. **Handover (2026-11-26):** `HANDOVER.md` written for readers with GitHub access only; the final
   results table is copied to `docs/results/`. Branches `gemmaOriginal` and `agent/google-drive-input`
   (both merged) were deleted. `live-memory-40` is committed and pushed, not merged into `main`.

## Still open

- [ ] **Off-server copy of `outputs/`:** the results behind the report's numbers are not in git;
      keep `backup/thesis_results_20261121.tgz` somewhere off the server.
