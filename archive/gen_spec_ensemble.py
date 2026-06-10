"""Mixed-feature-basis ensemble: 86-feat members + 86+spectral members -- genuine diversity.

Two prior lessons:
  1. More draws of the SAME 86 features (15-member + TTA) came back flat/worse -- that's
     correlated noise on one information basis, not new signal.
  2. Family-C spectral features (FFT entropy/centroid/band-energy, stable variant) showed
     REAL signal in the gate: mean +0.0075 (3x noise), tied on partition 0 (-0.0001,
     noise-level), and big robust gains on partitions 1/2 (+0.011 each). It missed the
     strict "never worse" bar by 0.0001 -- not absence of signal, just not a clean
     feature-set swap.

The fix: don't swap feature sets -- BLEND models that see different information. Half the
members train on the frozen 86 features, half on 86+spectral (134), all with the proven
augmentation (K=2 synthetic subjects/user, rot20/sm.1/ss.2), distinct seeds. Their errors
should be less correlated than two 86-feature draws -> real ensembling gain, mirroring the
0.7904->0.7958 jump from single-model -> same-basis-ensemble.

Writes submission_spec_ens.csv; reproducibility-checked (two refits identical).
Run: python gen_spec_ensemble.py
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
from stepFamilyC_spectral import family_C_features
from stepAugTTA import TUNED, K_AUG, ROT, SM, SS, transform, uid

N_86, N_SPEC = 5, 5            # members per feature basis
OUT = "/root/dm-assignment3/submission_spec_ens.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (L.SEED * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def build_basis(X_raw, basis, XC=None):
    """basis='86' -> just family A+B; basis='spec' -> A+B concat with precomputed spectral."""
    X86 = build_features(X_raw)
    if basis == "86":
        return X86
    return np.hstack([X86, XC if XC is not None else family_C_features(X_raw)])


def ensemble_proba(X_raw, y, user, Xte_raw, X86_te, XC_te):
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(Xte_raw), L.N_CLASSES)); n = 0
    Xte_basis = {"86": X86_te, "spec": np.hstack([X86_te, XC_te])}

    for basis, n_mem, off in [("86", N_86, 0), ("spec", N_SPEC, 1000)]:
        for e in range(n_mem):
            member = off + e
            feats, labs = [], []
            for k in range(-1, K_AUG):                        # k=-1 -> real (unaugmented)
                Xr = X_raw if k < 0 else aug_member(X_raw, user, k, member)
                feats.append(build_basis(Xr, basis)); labs.append(y)
            Xtr, ytr = np.vstack(feats), np.concatenate(labs)
            p = dict(full); p["random_state"] = L.SEED + member
            m = lgb.LGBMClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
            proba += m.predict_proba(Xte_basis[basis]); n += 1
            print(f"  [{basis:4}] member {e} done")
    return proba / n


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    T, prior = L.estimate_transition(y, fid, user)

    print("Precomputing spectral features for test (cached across refits)...")
    X86_te = build_features(Xte_raw)
    XC_te = family_C_features(Xte_raw)

    print(f"\ntraining mixed ensemble: {N_86} on 86-feat + {N_SPEC} on 86+spectral (refit #1)")
    pr1 = ensemble_proba(X_raw, y, user, Xte_raw, X86_te, XC_te)
    p1 = L.decode_test(pr1, fid_te, user_te, T, prior, **L.CURRENT)

    print("\nrefit #2 (reproducibility check)")
    pr2 = ensemble_proba(X_raw, y, user, Xte_raw, X86_te, XC_te)
    p2 = L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)
    print(f"Reproducible: {np.array_equal(p1, p2)}")
    assert np.array_equal(p1, p2), "NOT reproducible -- check seeds!"

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
