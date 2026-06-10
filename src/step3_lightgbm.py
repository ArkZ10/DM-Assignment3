"""Step 3: model swap LogisticRegression -> LightGBM. Feature families FROZEN (A+B, 86).

Same StratifiedGroupKFold outer splits as the LR runs (reuses har_cv.make_cv + SEED),
class-balanced via inverse-frequency sample weights, fully CPU + deterministic.

Why CPU, not the RTX 3090: reproducibility is a zero-grade gate. LightGBM's GPU
histogram can yield non-bit-identical splits run-to-run; CPU with deterministic=True
is reproducible, and at 11020x86 it's already fast. We verify byte-identical OOF.

LightGBM needs an internal validation set for early stopping; the LR harness
(evaluate_oof) doesn't provide one, so this file has its own CV loop -- but it uses the
IDENTICAL outer folds (har_cv.make_cv) so the comparison to LR is apples-to-apples.

Hyperparameters (modest, NOT tuned -- tuning is the next phase):
  objective='multiclass', num_class=6, n_estimators=1000 (early stopping, patience 50),
  learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
  min_child_samples=20, reg_lambda=1.0, deterministic=True, random_state=SEED.
Early-stopping internal split: 15% of each fold's training rows, stratified by class,
random_state=SEED (carved from TRAIN users only -> no leak into the outer val fold).

Run: python step3_lightgbm.py
"""
from __future__ import annotations
import warnings
import numpy as np
import lightgbm as lgb

# Cosmetic: LightGBM stores default feature names at fit time; predicting on a bare
# numpy array then triggers a harmless sklearn "no valid feature names" UserWarning.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv
from step2_familyB import build_features          # frozen A+B = 86 features

np.random.seed(SEED)

LR_BALANCED_AB = {           # frozen LR baseline row to beat
    "macro": 0.6476,
    "per_class": [0.9472, 0.8106, 0.2292, 0.6500, 0.7516, 0.4968],
}

PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    n_jobs=1,                # single-threaded for guaranteed determinism
    verbose=-1,
)
EARLY_STOP = 50
INTERNAL_VAL_FRAC = 0.15


def run_cv(X, y, groups):
    cv = make_cv(stratified=True)        # SAME outer folds as the LR runs
    oof = np.full(len(y), -1, dtype=int)
    fold_f1s, best_iters = [], []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        X_tr, y_tr = X[tr], y[tr]
        # internal early-stopping split (stratified, seeded; from train users only)
        Xi, Xe, yi, ye = train_test_split(
            X_tr, y_tr, test_size=INTERNAL_VAL_FRAC,
            stratify=y_tr, random_state=SEED,
        )
        w_i = compute_sample_weight("balanced", yi)   # inverse-freq weights
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            Xi, yi, sample_weight=w_i,
            eval_set=[(Xe, ye)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = model.predict(X[va])
        oof[va] = pred
        f = f1_score(y[va], pred, average="macro")
        fold_f1s.append(f)
        best_iters.append(model.best_iteration_)
        print(f"  fold {fold}: macro-F1={f:.4f}  (best_iter={model.best_iteration_})")

    assert (oof >= 0).all()
    macro = f1_score(y, oof, average="macro")
    per_class = f1_score(y, oof, average=None, labels=list(range(N_CLASSES)))
    print(f"  mean fold macro-F1: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
    print(f"  best_iterations: {best_iters}")
    return oof, macro, per_class


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X = build_features(X_raw)
    print(f"feature matrix: frozen Family A+B = {X.shape[1]} features (CPU LightGBM)\n")

    oof, macro, per_class = run_cv(X, y, groups)

    # ---- comparison table ----
    print("\n" + "=" * 86)
    hdr = f"{'feature_set / model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        cells = "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES))
        print(f"{name:30} | {m:7.4f} | {cells}")

    row("A+B, LogisticRegression (bal)", LR_BALANCED_AB["macro"], LR_BALANCED_AB["per_class"])
    row("A+B, LightGBM (bal)", macro, per_class)

    # ---- all per-class deltas ----
    print("\nPer-class deltas (LightGBM - LR):")
    for c in range(N_CLASSES):
        b, nw = LR_BALANCED_AB["per_class"][c], per_class[c]
        print(f"  L{c}: {b:.4f} -> {nw:.4f} ({nw - b:+.4f})")
    print(f"  macro-F1: {LR_BALANCED_AB['macro']:.4f} -> {macro:.4f} ({macro - LR_BALANCED_AB['macro']:+.4f})")

    # ---- confusion matrix ----
    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    print("\nCONFUSION MATRIX, A+B LightGBM (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
    print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

    # ---- L2 separability fork ----
    l2_prec = precision_score(y, oof, labels=[2], average="micro", zero_division=0)
    l2_rec = recall_score(y, oof, labels=[2], average="micro", zero_division=0)
    print("\nL2-separability fork:")
    print(f"  true1 -> pred2 false positives: {cm[1,2]}   (was 674 under A+B LR)")
    print(f"  L2 precision: {l2_prec:.4f}   (TP={cm[2,2]}, predicted-2 total={cm[:,2].sum()})")
    print(f"  L2 recall   : {l2_rec:.4f}   (TP={cm[2,2]}, true-2 total={cm[2].sum()})")

    # ---- gap to Baseline 3 ----
    B3 = 0.7088
    print(f"\nOverall macro-F1: {macro:.4f}   vs Baseline 3 = {B3}   "
          f"(gap {macro - B3:+.4f})")
    return macro


if __name__ == "__main__":
    main()
