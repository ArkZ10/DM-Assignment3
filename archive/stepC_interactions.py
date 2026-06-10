"""STEP C -- explicit pairwise feature interactions (then full pipeline incl. smoother).

Take the top-10 features by LightGBM gain importance (full-data fit), add all C(10,2)=45
pairwise products, rerun the whole pipeline (tuned LGB + current smoother) across the 3
partitions. Keep only if it clears the robustness gate vs 0.7405 mean; else drop.

Run: python stepC_interactions.py
"""
from __future__ import annotations
import numpy as np
from itertools import combinations
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
import temporal_lib as L


def top10_feature_idx(X, y):
    sw = compute_sample_weight("balanced", y)
    m = lgb.LGBMClassifier(**L.TUNED)
    m.fit(X, y, sample_weight=sw)
    imp = m.booster_.feature_importance(importance_type="gain")
    return np.argsort(imp)[::-1][:10]


def add_interactions(X, idx):
    prods = [(X[:, a] * X[:, b])[:, None] for a, b in combinations(idx, 2)]
    return np.hstack([X] + prods)


def main():
    X, y, fid, user = L.load_train()
    groups = user

    print("Fitting full-data LGB to rank importances...")
    idx = top10_feature_idx(X, y)
    print(f"top-10 feature indices: {idx.tolist()}")
    Xi = add_interactions(X, idx)
    print(f"features: {X.shape[1]} -> {Xi.shape[1]} (+{Xi.shape[1]-X.shape[1]} interactions)\n")

    print("Base OOF probs across 3 partitions (original 86 feats)...")
    parts_base = L.make_partitions(X, y, groups)
    base_ms = L.eval_macro(parts_base, y, fid, user, groups, **L.CURRENT)

    print("Base OOF probs across 3 partitions (with interactions)...")
    parts_int = L.make_partitions(Xi, y, groups)
    int_ms = L.eval_macro(parts_int, y, fid, user, groups, **L.CURRENT)

    print("\n" + "=" * 60)
    print(f"86 feats     : mean={base_ms.mean():.4f} {[round(m,4) for m in base_ms]}")
    print(f"+interactions: mean={int_ms.mean():.4f} {[round(m,4) for m in int_ms]}")
    print(f"delta mean: {int_ms.mean()-base_ms.mean():+.4f}  (noise {L.NOISE})")
    win = L.robust_gate(int_ms, base_ms)
    print(f"ROBUST WIN (mean +>{L.NOISE} AND not worse on any partition): {win}")

    if win:
        print("\n>>> robust win -> writing submission_interactions.csv")
        Xtest_raw, _ = L.D.load_split("test")
        Xtest = L.build_features(Xtest_raw)
        Xtest_i = add_interactions(Xtest, idx)
        L.write_submission(Xi, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_interactions.csv", X_test=Xtest_i)
    else:
        print("\n>>> not a robust win -> drop interactions, keep submission_temporal.csv.")


if __name__ == "__main__":
    main()
