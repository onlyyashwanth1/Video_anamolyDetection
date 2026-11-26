# Results

The final numbers of the project, the same as in Chapter 5 of the report.

## How to read them

Every test video has frame-by-frame labels saying which frames are abnormal. Each model
gives every frame an anomaly score. The scores of **all test frames of a dataset are
pooled together** and compared with the labels:

- **AUC (ROC-AUC):** the chance that a randomly picked abnormal frame scores higher than
  a randomly picked normal frame. 50 = random guessing, 100 = perfect.
- **AP (average precision):** how clean the top of the ranking is: whether the
  highest-scoring frames really are the abnormal ones. Random guessing gives the
  share of abnormal frames, so AP is low on datasets where anomalies are rare
  (about 8% of frames in UCF-Crime).

Neither metric needs an alert threshold; they judge the ranking of the scores. As is
standard for each benchmark, XD-Violence is reported with AP and the other three with AUC.
All numbers are percentages.

## Main table

| Model | UCF-Crime (AUC) | XD-Violence (AP) | ShanghaiTech (AUC) | Avenue (AUC) |
|---|---|---|---|---|
| **B + J (proposed)** | 78.10 | 64.86 | **63.96** | 63.86 |
| B alone (no judge) | 72.58 | 54.00 | 62.17 | 61.27 |
| Action + memory (baseline) | 78.09 | 60.73 | 61.75 | **66.38** |
| Shared configuration (baseline) | **81.58** | **74.36** | 52.09 | 58.05 |

- **B + J** uses no text prompts and is the most even model across the four datasets.
  Adding the judge J to B improves every dataset (+5.5 AUC on UCF-Crime, +10.9 AP on
  XD-Violence).
- **Shared configuration** is the best on UCF-Crime and XD-Violence because its text prompts
  name those datasets' anomaly types (fighting, shooting, explosion, ...). On ShanghaiTech
  and Avenue, whose anomalies are different (cycling, running, throwing), it falls close
  to chance.
- **Action + memory** mixes text prompts (80%) with the normal memory (20%) and is the
  best on Avenue.

## Both metrics for every dataset

| Model | UCF-Crime AUC / AP | XD-Violence AUC / AP | ShanghaiTech AUC / AP | Avenue AUC / AP |
|---|---|---|---|---|
| **B + J (proposed)** | 78.10 / 24.69 | 87.65 / 64.86 | 63.96 / 56.81 | 63.86 / 44.91 |
| B alone (no judge) | 72.58 / 17.29 | 83.86 / 54.00 | 62.17 / 53.76 | 61.27 / 32.54 |
| Action + memory (baseline) | 78.09 / 19.61 | 85.82 / 60.73 | 61.75 / 54.51 | 66.38 / 47.64 |
| Shared configuration (baseline) | 81.58 / 25.99 | 91.55 / 74.36 | 52.09 / 46.16 | 58.05 / 33.36 |

## Datasets

| Dataset | Normal training videos | Test videos | Kind of anomaly |
|---|---|---|---|
| UCF-Crime | 800 | 290 | crimes and accidents in surveillance footage (13 types) |
| XD-Violence | 2046 | 800 | violence in films, sports and news (6 types) |
| ShanghaiTech | 330 | 107 | unusual behaviour on a university campus |
| CUHK Avenue | 16 | 21 | unusual behaviour in front of a building |

Of each dataset's normal videos, 80% go into the memory and 20% are held out to calibrate
the scores.

## Caveat

The prompts, weights and smoothing of the models were chosen while looking at benchmark
data (UCF-Crime and XD-Violence validation videos and ShanghaiTech), so these are
development results, not results on completely unseen data.

## Model names in the code

| Name here | Config file (`config/architectures/`) |
|---|---|
| B + J | `b_plus_j.yaml` |
| B alone | `memory_b.yaml` |
| Action + memory | `action_default.yaml` |
| Shared configuration | `shared_best.yaml` |
