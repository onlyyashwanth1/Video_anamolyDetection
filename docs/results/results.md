# Architecture results (generated 2026-11-21T18:11:35)

Pooled frame-level metrics from `outputs/evaluations/`, unmasked features, restructured code.
UCF / XD partitions: grouped 20% validation / 80% evaluation split (seed 42) of the earlier study.
The 80/20 baseline row re-scores the window components of the 95/5 action_default runs (`outputs/results/thesis_analyses_20261119/action_memory80_analyses.json`).

## Primary metric (UCF, ShanghaiTech, Avenue: ROC-AUC; XD: AP)

| Architecture | UCF AUC | XD AP | ShanghaiTech AUC | Avenue AUC |
|---|---|---|---|---|
| b_plus_j (B+J, proposed) | 78.10 (was 78.07, study) | 64.86 | 63.96 | 63.86 (was 63.8, study) |
| memory_b (B) | 72.58 (was 72.58, study) | 54.00 (was 54.0, study) | 62.17 (was 62.1, study) | 61.27 (was 61.3, study) |
| action + memory 80/20 (report baseline) | 78.09 | 60.73 | 61.75 | 66.38 |
| action_default (95/5 run) | 75.53 (was 77.57, masked) | 59.12 (was 53.41, masked) | 61.05 | 68.96 |
| shared_best | 81.58 (was 83.12, masked) | 74.36 (was 71.76, masked) | 52.09 (was 54.14, masked) | 58.05 (was 54.9, masked) |

## All metrics

| Dataset | Architecture | AUC all | AP all | AUC val | AP val | AUC eval | AP eval | Run |
|---|---|---|---|---|---|---|---|---|
| ucf | action_default (95/5 run) | 75.53 | 18.70 | 60.92 | 12.60 | 80.29 | 23.68 | `outputs/evaluations/ucf/action_default/20261118T201639282837Z` |
| ucf | shared_best | 81.58 | 25.99 | 76.80 | 11.35 | 83.07 | 32.58 | `outputs/evaluations/ucf/shared_best/20261118T202359156984Z` |
| ucf | memory_b (B) | 72.58 | 17.29 | 78.20 | 31.56 | 71.65 | 15.05 | `outputs/evaluations/ucf/memory_b/20261118T201426043299Z` |
| ucf | b_plus_j (B+J, proposed) | 78.10 | 24.69 | 78.06 | 27.02 | 79.50 | 25.60 | `outputs/evaluations/ucf/b_plus_j/20261118T191404164264Z` |
| xd | action_default (95/5 run) | 81.81 | 59.12 | 82.00 | 54.00 | 81.84 | 60.47 | `outputs/evaluations/xd/action_default/20261118T203831485142Z` |
| xd | shared_best | 91.55 | 74.36 | 94.79 | 76.75 | 90.69 | 73.90 | `outputs/evaluations/xd/shared_best/20261118T212817986003Z` |
| xd | memory_b (B) | 83.86 | 54.00 | 84.18 | 44.91 | 83.76 | 56.10 | `outputs/evaluations/xd/memory_b/20261118T211928287243Z` |
| xd | b_plus_j (B+J, proposed) | 87.65 | 64.86 | 88.73 | 62.45 | 87.47 | 65.82 | `outputs/evaluations/xd/b_plus_j/20261118T213600396193Z` |
| shanghaitech | action_default (95/5 run) | 61.05 | 52.68 | — | — | — | — | `outputs/evaluations/shanghaitech/action_default/20261118T203843313025Z` |
| shanghaitech | shared_best | 52.09 | 46.16 | — | — | — | — | `outputs/evaluations/shanghaitech/shared_best/20261118T204006814420Z` |
| shanghaitech | memory_b (B) | 62.17 | 53.76 | — | — | — | — | `outputs/evaluations/shanghaitech/memory_b/20261118T203943367587Z` |
| shanghaitech | b_plus_j (B+J, proposed) | 63.96 | 56.81 | — | — | — | — | `outputs/evaluations/shanghaitech/b_plus_j/20261118T204030322562Z` |
| avenue | action_default (95/5 run) | 68.96 | 49.88 | — | — | — | — | `outputs/evaluations/avenue/action_default/20261118T210155594149Z` |
| avenue | shared_best | 58.05 | 33.36 | — | — | — | — | `outputs/evaluations/avenue/shared_best/20261118T210240452595Z` |
| avenue | memory_b (B) | 61.27 | 32.54 | — | — | — | — | `outputs/evaluations/avenue/memory_b/20261118T210219336443Z` |
| avenue | b_plus_j (B+J, proposed) | 63.86 | 44.91 | — | — | — | — | `outputs/evaluations/avenue/b_plus_j/20261118T210301428719Z` |
| ucf | action + memory 80/20 (report baseline) | 78.09 | 19.61 | 69.08 | 15.68 | 81.24 | 22.25 | `outputs/evaluations/ucf/action_default/20261118T201639282837Z` |
| xd | action + memory 80/20 (report baseline) | 85.82 | 60.73 | 85.95 | 56.05 | 85.85 | 62.03 | `outputs/evaluations/xd/action_default/20261118T203831485142Z` |
| shanghaitech | action + memory 80/20 (report baseline) | 61.75 | 54.51 | — | — | — | — | `outputs/evaluations/shanghaitech/action_default/20261118T203843313025Z` |
| avenue | action + memory 80/20 (report baseline) | 66.38 | 47.64 | — | — | — | — | `outputs/evaluations/avenue/action_default/20261118T210155594149Z` |
