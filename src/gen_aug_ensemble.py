"""Augmentation-seed ENSEMBLE -- variance reduction over synthetic-subject draws.

We saw single augmentation realizations swing the LB ~0.02 (det 0.77 vs hash 0.7904).
The fix: train E augmented LGBs, each with a DIFFERENT synthetic-subject draw, average
their test probabilities, then HMM. Averaging cancels the lucky/unlucky realization noise
-> a result that converges to the true mean and is much more LB-stable than one draw.

Writes submission_aug_ens.csv; reproducibility-checked. Same winning strength
(K=2 synthetic subjects/user, rot20/sm.1/ss.2), deterministic seeds.

Run: python gen_aug_ensemble.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

import har_data as D
import temporal_lib as L
from step2_familyB import build_features
from stepAugTTA import TUNED, K_AUG, ROT, SM, SS, transform, uid

E = 5                                  # ensemble members (distinct augmentation draws)
OUT = "/root/dm-assignment3/submission_aug_ens.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (L.SEED * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)
    full = dict(TUNED); full["n_estimators"] = 463

    def ensemble_proba():
        proba = np.zeros((len(X86_te), L.N_CLASSES))
        for e in range(E):
            feats, labs = [X86], [y]
            for k in range(K_AUG):
                feats.append(build_features(aug_member(X_raw, user, k, e))); labs.append(y)
            Xtr, ytr = np.vstack(feats), np.concatenate(labs)
            m = lgb.LGBMClassifier(**full)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            proba += m.predict_proba(X86_te)
        return proba / E

    print(f"training {E}-member augmentation ensemble (refit #1)...")
    pr1 = ensemble_proba()
    p1 = L.decode_test(pr1, fid_te, user_te, T, prior, **L.CURRENT)
    print("refit #2 (reproducibility)...")
    pr2 = ensemble_proba()
    p2 = L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)
    assert np.array_equal(p1, p2), "NOT reproducible!"
    print("Reproducibility: two ensembles -> identical  [OK]")

    sample = pd.read_csv(SUB_TEMPLATE)
    pbi = dict(zip(fid_te.tolist(), p1.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print("Label dist:", {c: Counter(out.tolist()).get(c, 0) for c in range(L.N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
