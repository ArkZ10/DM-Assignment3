"""STEP C -- F1-driven feature selection on the A+B feature set (tuned LightGBM).

Goal: prune the 86 features to the subset that maximizes (or holds) OOF macro-F1, for
a simpler, more reproducible model. Procedure:
  1. Rank features by LightGBM gain importance, averaged over the 5 CV folds (computed
     leak-free: importance from each fold's train model only).
  2. Backward/forward sweep over top-K prefixes (K = 10..86), scoring each K with the
     full StratifiedGroupKFold OOF macro-F1 on the tuned config.
  3. Keep the smallest K whose OOF macro-F1 is >= full-set score (simpler = better);
     otherwise keep the K with the best OOF macro-F1.

Run: python step7_selectC.py
"""
from __future__ import annotations
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv
from lgbm_cv import BASE_PARAMS, lgbm_oof, INTERNAL_VAL_FRAC
from sklearn.utils.class_weight import compute_sample_weight
from step2_familyB import build_features

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

STEP_A = {"macro": 0.7236,
          "per_class": [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]}

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)
N_FEAT = X.shape[1]


def fold_importance_ranking():
    """Mean gain importance over folds (each fold's importance from its train model)."""
    cv = make_cv(stratified=True)
    imp = np.zeros(N_FEAT)
    for tr, _ in cv.split(X, y, groups):
        X_tr, y_tr = X[tr], y[tr]
        Xi, Xe, yi, ye = train_test_split(
            X_tr, y_tr, test_size=INTERNAL_VAL_FRAC, stratify=y_tr, random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**TUNED)
        m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        imp += m.booster_.feature_importance(importance_type="gain")
    return np.argsort(imp)[::-1]   # indices, most important first


def main():
    print("STEP C: F1-driven feature selection (tuned LightGBM). Full set = 86 feats.\n")
    order = fold_importance_ranking()

    Ks = [10, 15, 20, 25, 30, 40, 50, 60, 70, 86]
    print(f"{'K':>4} | {'OOF macroF1':>11} | {'fold_std':>8} | L2     L5")
    print("-" * 50)
    results = []
    for K in Ks:
        cols = order[:K]
        _, macro, pc, fs, _ = lgbm_oof(X[:, cols], y, groups, params=TUNED)
        results.append((K, macro, np.std(fs), pc))
        print(f"{K:>4} | {macro:>11.4f} | {np.std(fs):>8.4f} | {pc[2]:.4f} {pc[5]:.4f}"
              + ("   <- full set" if K == N_FEAT else ""))

    full_macro = [r for r in results if r[0] == N_FEAT][0][1]
    # smallest K whose macro >= full set (within 1e-4); else best macro
    at_least_full = [r for r in results if r[1] >= full_macro - 1e-4 and r[0] < N_FEAT]
    if at_least_full:
        chosen = min(at_least_full, key=lambda r: r[0])
        reason = f"smallest K with OOF >= full-set ({full_macro:.4f})"
    else:
        chosen = max(results, key=lambda r: r[1])
        reason = "best OOF macro-F1"
    K, macro, std, pc = chosen

    print("\n" + "=" * 86)
    hdr = f"{'model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def prow(name, m, p):
        print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{p[c]:.4f}" for c in range(N_CLASSES)))

    prow("Step A: full 86 feats", STEP_A["macro"], STEP_A["per_class"])
    prow(f"Step C: top-{K} feats", macro, pc)
    print(f"\nChosen K={K} ({reason}). macro {macro:.4f} vs full {full_macro:.4f} "
          f"({macro - full_macro:+.4f}), fold_std {std:.4f}")
    print(f"L2: {STEP_A['per_class'][2]:.4f} -> {pc[2]:.4f} ({pc[2]-STEP_A['per_class'][2]:+.4f})  "
          f"L5: {STEP_A['per_class'][5]:.4f} -> {pc[5]:.4f} ({pc[5]-STEP_A['per_class'][5]:+.4f})")

    # persist selected columns if we actually prune
    if K < N_FEAT:
        json.dump([int(c) for c in order[:K]],
                  open("/root/dm-assignment3/selected_features.json", "w"))
        print(f"\nSaved {K} selected feature indices to selected_features.json")
    else:
        print("\nNo pruning beat the full set -> keep all 86 features.")


if __name__ == "__main__":
    main()
