# Pending decisions

Things that need your call. Claude keeps this file updated: tick `[x]` and add a line
when you decide, or just tell Claude in chat.

Live results table: `outputs/results/architectures/results.md`
(refresh with `.venv/bin/python scripts/collect_results.py`).

## Open

### 1. Should the thesis's proposed method drop object masking? (2026-11-19)
The restructured code has no masking, so its numbers for the action models are lower on UCF:

| | masked (thesis today) | unmasked (new code) |
|---|---|---|
| shared_best, UCF AUC | 83.12 | 81.58 (= the thesis's own masking ablation, Table 5.x `tab:masking`) |
| action_default, UCF AUC | 77.57 | 75.53 |

XD is still running (the ablation predicts shared_best ≈ 74.36 AP unmasked, higher than masked 71.76).

The thesis presents the masked 95/5 pipeline as the proposed method: stage 2 of Chapter 4,
the abstract (77.57 / 53.41), Chapter 1's contributions, Chapter 5's main tables.

- **What Claude does without waiting (additive only):** a new Chapter 5 section with the
  re-run of all four architectures on the restructured code (UCF, XD, ShanghaiTech, Avenue),
  plus docs/ARCHITECTURES.md. Nothing existing is deleted.
- [ ] **A.** Rewrite around the code that ships: remove stage 2 from Chapter 4, make the
  unmasked numbers the headline, keep masking only as the ablation (`sec:masking-ablation`).
- [ ] **B.** Keep the thesis as is (masked pipeline = what was evaluated in Chapters 4–5) and
  present the restructure as later work in the new section only. (Claude's recommendation if
  the submission is soon: least risk, still honest.)
- [ ] **C.** Make B+J the proposed method (it generalises best: Avenue 63.8 vs 54.9) and move the
  action pipeline to a baseline chapter. Largest rewrite.

**Decided 2026-11-19: A — done.** The thesis (`BTP_THESIS_VAD/`, 81 pages) no longer mentions
masking: Ch1/2/4/6, appendices, Figure 4.1 (stages renumbered 1–8) rewritten, and every number
re-measured without masking — the four architectures on four datasets in `outputs/evaluations/`,
the older studies re-run from cached features in `outputs/results/thesis_analyses_20261119/`.
Backup of the masked version: `BTP_THESIS_VAD_backup_masked_20261119.tgz`.

### 1c. B+J is now the proposed model (decided 2026-11-19, done)
Ch4 presents B+J (11 stages, same numbering as the 'Memory + Judge' site; new Figure 4.1) with
the default model and the shared configuration as baselines (Sec. 4.8, own figure). Ch5 leads
with B+J: results on 4 datasets, ablation, sensitivity, alert behaviour; qualitative examples and
the XD error audit are now for B+J. Abstract, Ch1 (objectives, contributions) and Ch6 reframed.
89-page PDF. Stated caveats: B+J settings were chosen on UCF/XD validation + ShanghaiTech;
on clean Avenue the default baseline is better (68.96 vs 63.86); B+J is offline and expensive.
Backup of the previous version (default model proposed): `BTP_THESIS_VAD_backup_default_proposed_20261119.tgz`.

### 1b. Please read: the numbers change some conclusions (2026-11-19)
The thesis text now follows these numbers (UCF AUC / XD AP / ShanghaiTech AUC / Avenue AUC):

| Model | UCF | XD | SHT | Avenue (clean) |
|---|---|---|---|---|
| default 95/5 | 75.53 | 59.12 | 61.05 | **68.96** |
| shared configuration | **81.58** | **74.36** | 52.09 | 58.05 |
| B | 72.58 | 54.00 | 62.17 | 61.27 |
| B+J | 78.10 | 64.86 | **63.96** | 63.86 |

- **Avenue:** the default model is best (its action prompts alone: 69.2). The old claim "B+J is
  best on the untouched Avenue test" is gone; B+J is now "most balanced, best prompt-free,
  best on ShanghaiTech".
- **XD B+J is now complete** (64.86 AP, was never finished before) and beats the default model.
- **Default model:** memory alone is strong (UCF 71.2, XD 43.4 AP); the weight sweep peaks at
  w_a = 0.8 and the validation search picks 0.73. The thesis keeps 95/5 as the fixed reference
  and says it under-weights memory.
- **Threshold calibration** works on XD (F1 38.5 → 57.9) but fails on UCF (validation half
  unrepresentative); the thesis says so.
- **Prompt prior:** category prompts are worth ~7 UCF AUC and ~19 XD AP points (was ~6 / ~12).
- [ ] OK with this story / [ ] want a different framing: …

### 2. The first walkthrough site still shows masking (2026-11-19)
https://claude.ai/artifact/9MNzLJzMo59vms754oeoig has "② Object-centric masking" as a step
and in the diagram. If you choose 1A or 1C, Claude should remove that step (and redraw the
draw.io diagram without it). Not touched until you decide.
- [ ] Remove masking from the first site
- [ ] Leave it

### 3. New B+J walkthrough site: sharing (2026-11-19)
Published privately: https://claude.ai/artifact/YUqrMTZtJ3iU8gGB5jaQW3 ("Memory + Judge",
12 steps, same look as the first site). Only you can open it until you share it from the
page's Share menu. Sources: `docs/site/memory_judge_src/` (rebuild: `python docs/site/memory_judge_src/build.py`).
Following your rules for the first site it has no dataset names, no benchmark numbers and no
online/offline wording. Note that B+J really does need the whole video (centring uses all
captions, smoothing looks ahead); say if you want that stated on the site.
- [ ] Looks right / share it
- [ ] Changes wanted: …

### 4. Git: push / merge `restructure` (2026-11-19)
`restructure` (3877d03 + the commit Claude will make for results/docs) is local only.
Claude will commit locally but will not push or merge without you.
- [ ] Push `restructure` to origin
- [ ] Merge into `main`

