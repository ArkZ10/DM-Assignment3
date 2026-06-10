"""Generate submission for the WINNING config: augmented LightGBM + HMM smoother.

Augmented LGB+HMM cleared the fair 3-partition gate (OOF 0.7341 vs base 0.7287, better on
all 3 partitions). This refits on ALL 60 users + K synthetic copies each, predicts the
6849 test files, HMM-smooths per user, and writes submission_aug.csv. Verifies Id order,
label range, distribution, and reproducibility (two refits -> identical).

Run: python gen_aug_submission.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

import har_data as D
import temporal_lib as L
from step2_familyB import build_features
from lgbm_cv import BASE_PARAMS
from stepAug import augment_subset, K_AUG

TUNED = dict(BASE_PARAMS); TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
TUNED["n_estimators"] = 463          # median CV best-iter (no early stopping on full refit)
OUT = "/root/dm-assignment3/submission_aug.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)

    Xte_raw, meta_te = D.load_split("test")
    assert "label" not in meta_te.columns
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)

    T, prior = L.estimate_transition(y, fid, user)

    def fit_and_predict():
        feats = [X86]; labs = [y]
        for k in range(K_AUG):
            Xaug = augment_subset(X_raw, user, L.SEED, k)     # all users get synthetic copies
            feats.append(build_features(Xaug)); labs.append(y)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        sw = compute_sample_weight("balanced", ytr)
        model = lgb.LGBMClassifier(**TUNED)
        model.fit(Xtr, ytr, sample_weight=sw)
        probs = model.predict_proba(X86_te)
        return L.decode_test(probs, fid_te, user_te, T, prior, **L.CURRENT)

    print("refit #1 (all users + synthetic copies)...")
    p1 = fit_and_predict()
    print("refit #2 (reproducibility check)...")
    p2 = fit_and_predict()
    assert np.array_equal(p1, p2), "augmented refit NOT reproducible!"
    print("Reproducibility: two refits -> identical predictions  [OK]")

    pred_by_id = dict(zip(fid_te.tolist(), p1.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()), "Id mismatch!"
    assert len(sample) == 6849 and set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)

    dist = Counter(out.tolist())
    print(f"Verify: Ids match template={set(pred_by_id)==set(sample['Id'].tolist())}, "
          f"rows={len(sample)}, labels in 0-5={set(np.unique(out)).issubset(set(range(6)))}")
    print("Label distribution:", {c: dist.get(c, 0) for c in range(L.N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
