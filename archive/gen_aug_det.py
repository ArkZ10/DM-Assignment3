"""Regenerate the augmentation submission with DETERMINISTIC seeding (reproducible).

Same winning method (K=2 synthetic subjects/user, rot20/sm.1/ss.2) + HMM, but using the
deterministic per-user augmentation (fixed seeding, not Python's salted hash). Clean OOF
0.7386 (3 distinct partitions) vs the hash-based 0.7341 behind submission_aug.csv (LB
0.7904). Writes submission_aug_det.csv; verifies reproducibility (two refits identical).

Run: python gen_aug_det.py
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
from stepAugTTA import TUNED, K_AUG, aug_train          # deterministic-uid augmentation

OUT = "/root/dm-assignment3/submission_aug_det.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)
    full = dict(TUNED); full["n_estimators"] = 463

    def fit_pred():
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_train(X_raw, user, k))); labs.append(y)
        Xtr, ytr = np.vstack(feats), np.concatenate(labs)
        m = lgb.LGBMClassifier(**full)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        return L.decode_test(m.predict_proba(X86_te), fid_te, user_te, T, prior, **L.CURRENT)

    print("refit #1 ..."); p1 = fit_pred()
    print("refit #2 (reproducibility) ..."); p2 = fit_pred()
    assert np.array_equal(p1, p2), "NOT reproducible!"
    print("Reproducibility: two refits -> identical  [OK]")

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
