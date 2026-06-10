"""Diverse augmentation ensemble -- push the lever that just gave +0.0054 LB (0.7958).

Ensembling diverse augmented models adds real LB value. This maximizes diversity across
THREE axes, all averaged then HMM-smoothed:
  - model type: LightGBM + XGBoost + CatBoost (decorrelated learners)
  - augmentation strength: rot20/sm.1/ss.2 and rot30/sm.15/ss.3 (varied synthetic subjects)
  - random seed: distinct synthetic-subject draws per member
All members trained on real + K synthetic subjects. Deterministic, reproducibility-checked.

Run: python gen_mega_ensemble.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

import har_data as D
import temporal_lib as L
from step2_familyB import build_features
from stepAugTTA import TUNED, K_AUG, transform, uid

N_CLASSES = L.N_CLASSES
OUT = "/root/dm-assignment3/submission_mega.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"

# (strength) configs cycled across seeds
STRENGTHS = [(20.0, 0.10, 0.20), (30.0, 0.15, 0.30)]
N_LGB, N_XGB, N_CAT = 8, 3, 3        # members per model type

XGB_P = dict(objective="multi:softprob", num_class=N_CLASSES, n_estimators=500,
             learning_rate=0.0366, max_depth=6, subsample=0.7, colsample_bytree=0.73,
             reg_lambda=0.5, reg_alpha=0.75, min_child_weight=5, gamma=0.1,
             tree_method="hist", n_jobs=1, verbosity=0)
CAT_P = dict(loss_function="MultiClass", iterations=600, learning_rate=0.0366, depth=6,
             l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8, thread_count=1,
             verbose=0)


def aug_member(X_raw, user, k, member, deg, sm, ss):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (L.SEED * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), deg, sm, ss)
    return out


def aug_design(X_raw, X86, y, user, member, deg, sm, ss):
    feats, labs = [X86], [y]
    for k in range(K_AUG):
        feats.append(build_features(aug_member(X_raw, user, k, member, deg, sm, ss))); labs.append(y)
    return np.vstack(feats), np.concatenate(labs)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)
    lgb_p = dict(TUNED); lgb_p["n_estimators"] = 463

    def ensemble_proba():
        proba = np.zeros((len(X86_te), N_CLASSES)); n = 0
        for e in range(N_LGB):
            deg, sm, ss = STRENGTHS[e % len(STRENGTHS)]
            Xtr, ytr = aug_design(X_raw, X86, y, user, e, deg, sm, ss)
            p = dict(lgb_p); p["random_state"] = L.SEED + e
            m = lgb.LGBMClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            proba += m.predict_proba(X86_te); n += 1
            print(f"  LGB member {e} (rot{deg}) done")
        for e in range(N_XGB):
            deg, sm, ss = STRENGTHS[e % len(STRENGTHS)]
            Xtr, ytr = aug_design(X_raw, X86, y, user, 100 + e, deg, sm, ss)
            p = dict(XGB_P); p["random_state"] = L.SEED + e
            m = xgb.XGBClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            proba += m.predict_proba(X86_te); n += 1
            print(f"  XGB member {e} done")
        for e in range(N_CAT):
            deg, sm, ss = STRENGTHS[e % len(STRENGTHS)]
            Xtr, ytr = aug_design(X_raw, X86, y, user, 200 + e, deg, sm, ss)
            p = dict(CAT_P); p["random_seed"] = L.SEED + e
            m = CatBoostClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            proba += m.predict_proba(X86_te); n += 1
            print(f"  CAT member {e} done")
        return proba / n

    print(f"training diverse ensemble: {N_LGB} LGB + {N_XGB} XGB + {N_CAT} CAT (refit #1)")
    pr1 = ensemble_proba(); p1 = L.decode_test(pr1, fid_te, user_te, T, prior, **L.CURRENT)
    print("refit #2 (reproducibility)")
    pr2 = ensemble_proba(); p2 = L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)
    print(f"Reproducible: {np.array_equal(p1, p2)}")

    sample = pd.read_csv(SUB_TEMPLATE)
    pbi = dict(zip(fid_te.tolist(), p1.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print("Label dist:", {c: Counter(out.tolist()).get(c, 0) for c in range(N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
