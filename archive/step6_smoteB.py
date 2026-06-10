"""STEP B -- SMOTE / minority oversampling on the ENGINEERED features (not raw seq).

SMOTE is fit INSIDE each fold on training rows ONLY (the lgbm_cv harness applies the
resampler after the internal split, never to validation) -- no synthetic leakage.

Three regimes compared on the tuned (Step A) LightGBM:
  (1) class weights only          [current best, Step A]
  (2) SMOTE only (no weights)
  (3) SMOTE + weights             (weights recomputed on the SMOTE-balanced set, so
                                   they are naturally LIGHTER than on raw imbalanced data)
Prior evidence: sampling helps NNs more than trees, so this may not beat Step A. If it
doesn't, we say so and drop it.

Run: python step6_smoteB.py
"""
from __future__ import annotations
import json
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof
from step2_familyB import build_features

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

STEP_A = {"macro": 0.7236,
          "per_class": [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]}

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)

MINORITY = [2, 4, 5]   # rare classes to oversample


def make_smote_resampler(y_all):
    """Oversample minority classes 2,4,5 up toward the median majority count.

    Returns a resampler(Xi, yi) that fits SMOTE on the passed (train) rows only.
    Target per minority class = max(its count, target_n) where target_n is a moderate
    level (not full balance -- avoids drowning trees in synthetic points).
    """
    def resampler(Xi, yi):
        counts = Counter(yi)
        # moderate target: lift each minority class to ~half the largest class
        target_n = int(0.5 * max(counts.values()))
        strategy = {c: max(counts[c], target_n) for c in MINORITY if counts.get(c, 0) > 0}
        k = min(5, min(counts[c] for c in MINORITY if counts.get(c, 0) > 0) - 1)
        k = max(1, k)
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=SEED)
        return sm.fit_resample(Xi, yi)
    return resampler


def row(name, m, pc):
    print(f"{name:34} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))


def detail(tag, oof):
    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    p2 = precision_score(y, oof, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, oof, labels=[2], average="micro", zero_division=0)
    p5 = precision_score(y, oof, labels=[5], average="micro", zero_division=0)
    r5 = recall_score(y, oof, labels=[5], average="micro", zero_division=0)
    print(f"  [{tag}] L2 prec={p2:.4f} rec={r2:.4f} | L5 prec={p5:.4f} rec={r5:.4f} | "
          f"true1->pred2={cm[1,2]}")


def main():
    print("STEP B: SMOTE regimes on tuned LightGBM. SMOTE fit inside-fold, train-only.\n")
    resampler = make_smote_resampler(y)

    print("Running regime (1) class weights only [= Step A] ...")
    oof1, m1, pc1, f1s1, _ = lgbm_oof(X, y, groups, params=TUNED, use_weights=True)
    print("Running regime (2) SMOTE only, no weights ...")
    oof2, m2, pc2, f1s2, _ = lgbm_oof(X, y, groups, params=TUNED, use_weights=False,
                                      resampler=resampler)
    print("Running regime (3) SMOTE + class weights ...")
    oof3, m3, pc3, f1s3, _ = lgbm_oof(X, y, groups, params=TUNED, use_weights=True,
                                      resampler=resampler)

    print("\n" + "=" * 92)
    hdr = f"{'regime':34} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))
    row("Step A: weights only", STEP_A["macro"], STEP_A["per_class"])
    row("(1) weights only [re-run check]", m1, pc1)
    row("(2) SMOTE only", m2, pc2)
    row("(3) SMOTE + weights", m3, pc3)

    print("\nL2 / L5 precision+recall per regime:")
    detail("1 weights", oof1); detail("2 SMOTE", oof2); detail("3 SMOTE+w", oof3)

    print("\nPer-fold spread (std):")
    for tag, fs in [("1 weights", f1s1), ("2 SMOTE", f1s2), ("3 SMOTE+w", f1s3)]:
        print(f"  {tag}: {[round(f,4) for f in fs]}  std={np.std(fs):.4f}")

    print("\nDeltas vs Step A (0.7236):")
    for tag, m in [("(2) SMOTE only", m2), ("(3) SMOTE+weights", m3)]:
        print(f"  {tag}: {m:.4f} ({m - STEP_A['macro']:+.4f})")


if __name__ == "__main__":
    main()
