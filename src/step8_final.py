"""STEP D (final config) + STEP E (inference / submission).

FINAL config (locked):
  - Features: full A+B set (86 features), the highest-demonstrated-OOF set.
    (Step B SMOTE dropped: -0.0156. Step C top-50: wash within noise, did not beat full.)
  - Model: tuned LightGBM (Step A, tuned_params.json) + inverse-frequency sample weights.
  - No ensemble: the raw-sequence CNN (0.659) was far weaker; no measured blend lift.

D: report final OOF macro-F1, per-class F1, confusion matrix, per-fold spread, gap vs
   Baseline 3 (0.7088).
E: refit on ALL 11020 train files (no holdout; n_estimators = median CV best_iter),
   apply the IDENTICAL feature pipeline to the 6849 test files, predict, and write
   submission.csv aligned to sample_submission.csv's Id order. Verify + reproduce twice.

Run: python step8_final.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof, fit_full
from step2_familyB import build_features

BASELINE3 = 0.7088
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission.csv"

FINAL_PARAMS = dict(BASE_PARAMS)
FINAL_PARAMS.update(json.load(open("/root/dm-assignment3/tuned_params.json")))


def step_D():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X = build_features(X_raw)

    print("STEP D -- final config OOF (tuned LightGBM, 86 feats, inverse-freq weights)\n")
    oof, macro, per_class, fold_f1s, best_iters = lgbm_oof(X, y, groups, params=FINAL_PARAMS)

    print(f"{'class':>6} | " + " | ".join(f"L{c}" for c in range(N_CLASSES)))
    print("F1     | " + " | ".join(f"{per_class[c]:.4f}" for c in range(N_CLASSES)))
    print(f"\nFINAL OOF macro-F1: {macro:.4f}")
    print(f"per-fold: {[round(f,4) for f in fold_f1s]}")
    print(f"per-fold mean {np.mean(fold_f1s):.4f}  std {np.std(fold_f1s):.4f}  "
          f"min {min(fold_f1s):.4f}  max {max(fold_f1s):.4f}")
    print(f"gap vs Baseline 3 ({BASELINE3}): {macro - BASELINE3:+.4f}")
    print(f"CV best_iterations: {best_iters} -> median {int(np.median(best_iters))}")

    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    print("\nCONFUSION MATRIX (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
    print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

    train_dist = Counter(y)
    return X, y, int(np.median(best_iters)), train_dist


def step_E(X_train, y_train, n_estimators, train_dist):
    print("\n" + "=" * 70)
    print("STEP E -- refit on ALL train, predict test, write submission\n")

    # fix n_estimators (no early stopping on full-data refit), keep all else tuned
    final_params = dict(FINAL_PARAMS); final_params["n_estimators"] = n_estimators

    def build_and_predict():
        model = fit_full(X_train, y_train, params=final_params, use_weights=True)
        X_test_raw, meta_test = D.load_split("test")
        assert "label" not in meta_test.columns, "test must have no label column"
        X_test = build_features(X_test_raw)             # IDENTICAL pipeline (label-free)
        preds = model.predict(X_test)
        return meta_test["file_id"].to_numpy(), preds

    # reproducibility: two independent refits must give identical predictions
    ids1, p1 = build_and_predict()
    ids2, p2 = build_and_predict()
    assert np.array_equal(ids1, ids2) and np.array_equal(p1, p2), "refit NOT reproducible!"
    print("Reproducibility: two independent refits -> identical test predictions  [OK]")

    pred_by_id = dict(zip(ids1.tolist(), p1.tolist()))

    # align to sample_submission Id order exactly
    sample = pd.read_csv(SUB_TEMPLATE)
    sample_ids = sample["Id"].to_numpy()
    out_labels = np.array([pred_by_id[i] for i in sample_ids], dtype=int)
    sub = pd.DataFrame({"Id": sample_ids, "Label": out_labels})

    # ---- verifications BEFORE writing ----
    assert set(pred_by_id) == set(sample_ids.tolist()), "Id mismatch vs template!"
    assert len(sub) == 6849, f"row count {len(sub)} != 6849"
    assert set(np.unique(out_labels)).issubset(set(range(6))), "labels outside 0-5!"
    print(f"Verify: set(my Ids)==set(sample Ids): {set(pred_by_id)==set(sample_ids.tolist())}")
    print(f"Verify: row count == 6849: {len(sub) == 6849}")
    print(f"Verify: all labels in 0-5: {set(np.unique(out_labels)).issubset(set(range(6)))}")

    pred_dist = Counter(out_labels.tolist())
    n_tr = sum(train_dist.values())
    print("\nPredicted vs train label distribution:")
    print(f"  {'label':>5} | {'pred #':>7} {'pred %':>7} | {'train %':>8}")
    for c in range(N_CLASSES):
        pc = pred_dist.get(c, 0)
        print(f"  {c:>5} | {pc:>7} {100*pc/len(sub):>6.1f}% | {100*train_dist[c]/n_tr:>7.1f}%")
    assert all(pred_dist.get(c, 0) > 0 for c in range(N_CLASSES)), "a class collapsed to zero!"
    print("  (all 6 classes have nonzero predictions -- no collapse)")

    sub.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(sub)} rows)")


if __name__ == "__main__":
    X, y, med_iter, train_dist = step_D()
    step_E(X, y, med_iter, train_dist)
