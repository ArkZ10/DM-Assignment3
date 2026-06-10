# Data Mining Assignment 3: Human Activity Recognition

**Name:** Yeftha Joshua Ezekiel
**Student ID:** 314540079

Cross-subject Human Activity Recognition (6-class) on wearable accelerometer
data, submitted to the NYCU Data Mining Assignment 3 Kaggle competition.

## Result Summary

| | Macro-F1 (Public LB) |
|---|---|
| Baseline 3 (competition reference) | 0.7088 |
| **Final submission (`submission_pseudo_aug.csv`)** | **0.8061** |
| **Improvement** | **+0.0973** |

The final pipeline is a tuned 86-feature LightGBM model with HMM (Viterbi)
temporal smoothing, cross-subject augmentation, and two-pass semi-supervised
pseudo-labeling (Pass 1: 5-member ensemble on real data only -> pseudo-label
the 6,849 test files at confidence > 0.82 -> Pass 2: 7-member ensemble
retrained on real + augmented pseudo-labeled data).

LB progression from the tuned 86-feature baseline to the final submission:

| Stage | LB Macro-F1 | Delta |
|---|---|---|
| Tuned LGB (86 feat.) + HMM smoothing | 0.7867 | -- |
| + Cross-subject augmentation | 0.7904 | +0.0037 |
| + Augmentation-seed ensemble | 0.7958 | +0.0054 |
| + Pseudo-labeling (2-pass, threshold 0.82) | 0.8044 | +0.0086 |
| + Pseudo-label augmentation + 7-member Pass 2 (**final**) | **0.8061** | +0.0017 |

## Repository Structure

```
.
├── main.py                  # entry point: reproduces the final submission
├── src/                      # all source code (37 modules)
│   ├── gen_pseudo_aug.py     # final pipeline (LB 0.8061)
│   ├── har_data.py            # data loading
│   ├── step2_familyA.py,      # feature engineering (Family A/B)
│   │   step2_familyB.py, ...
│   ├── temporal_lib.py,       # HMM transition estimation + decoding
│   │   step13_temporal.py, decoder_lib.py
│   ├── stepAugTTA.py          # cross-subject augmentation + tuned params
│   ├── make_figures.py        # regenerates all figures in figures/
│   ├── audit.py                # preliminary data analysis (Section 2)
│   └── ...                     # remaining model-progression / ablation scripts (Sections 3-4)
├── archive/                  # superseded experiments and intermediate submissions
├── cache/                    # cached features/embeddings used by some scripts
├── figures/                  # figures used in report.pdf
├── nycu-data-mining-assignment-3/  # competition data (train/test, sample_submission.csv) ---> Need to be uploaded here
├── submission_pseudo_aug.csv      # final submission (LB 0.8061)
└── tuned_params.json          # Optuna-tuned LightGBM hyperparameters
```

## How to Run

### 1. Requirements

- Python 3.12
- numpy, pandas, scipy, scikit-learn, lightgbm, torch

Install with:

```bash
pip install numpy pandas scipy scikit-learn lightgbm torch
```

### 2. Data

The competition data must be present at
`nycu-data-mining-assignment-3/` (Need to be uploaded to this repo root), containing
the `train/`, `test/` directories and `sample_submission.csv`. No other setup
is required — all paths used by the code are relative to the repo root.

### 3. Run the pipeline

From the repo root, simply run:

```bash
python main.py
```

This is the single entry point and reproduces the top-scoring submission
end-to-end. Internally it calls `src/gen_pseudo_aug.py`, which:

1. Loads the train (11,020 files) and test (6,849 files) data and builds the
   86-feature set (Family A + Family B).
2. Estimates the per-user HMM transition matrix for temporal decoding.
3. **Pass 1**: trains a 5-member LightGBM ensemble (with cross-subject
   augmentation) on the real training data and predicts on the test set.
4. **Pseudo-labeling**: test files where the max class probability exceeds
   0.82 (5,756 / 6,849, 84.0%) are kept as pseudo-labeled training rows.
5. **Pass 2**: trains a 7-member LightGBM ensemble on real + augmented
   pseudo-labeled data and predicts on the test set.
6. Applies Viterbi temporal decoding to the Pass 2 probabilities.
7. Re-runs steps 3–6 a second time and asserts the predictions are
   bit-identical (`Reproducible: True`) before writing the output.

**Output:** `submission_pseudo_aug.csv` is (re)written at the repo root in the
format expected by Kaggle (`Id,Label`). This is the file to upload to the
competition leaderboard to reproduce the **0.8061 Macro-F1** score.

**Runtime:** roughly 5–10 minutes on CPU (it trains 24 LightGBM models total,
since the full Pass 1 + Pass 2 pipeline runs twice for the reproducibility
check).

### 4. (Optional) Regenerate the report figures

```bash
python src/make_figures.py
```

Writes all 10 figures used in report to `figures/`.
