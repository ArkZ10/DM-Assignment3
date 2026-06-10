"""Pseudo-labeling v4: stronger pass-1 (10 members) + more pseudo-aug (K=2).

Results so far:
  gen_pseudolabel_ens.py : 5+5,  thresh=0.82, K_ps=0 -> LB 0.8044
  gen_pseudo_aug.py      : 5+7,  thresh=0.82, K_ps=1 -> LB 0.8061
  gen_pseudo_v3.py       : 5+7+9 (3-pass)            -> WORSE (error compounding)

Pattern: 2 passes good, 3+ passes bad (compounding ~3% pseudo-label error).
         K_ps=1 over K_ps=0 gave +0.0017 (more test-distribution data).

This version maximises the 2-pass approach:
  Pass 1: 10 members (vs 5) -> stronger initial ensemble
    -> pseudo-labels are more accurate (better calibrated)
    -> MORE files cross the 0.82 threshold (more confident borderline cases)
  Pass 2: 10 members, K_AUG_PSEUDO=2 (vs 7 + K=1)
    -> 3x pseudo data per member vs K=0 original

At K_AUG_PSEUDO=2, pass-2 data per member (per row):
  real:   11020 * 3 = 33060
  pseudo: N_pseudo * 3 (1 original + 2 aug)
Total: ~50k+ rows per member -> substantial test-distribution coverage.

Writes submission_pseudo_v4.csv; reproducible.
Run: python gen_pseudo_v4.py
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

THRESH = 0.82
K_AUG_PSEUDO = 2
N1, N2 = 10, 10
PSEUDO_SEED = 777777
OUT = "/root/dm-assignment3/submission_pseudo_v4.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member, seed_base=L.SEED):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (seed_base * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def run_pass(X_raw, X86, y, user, X86_te, n_members, seed_offset,
             Xte_raw_ps=None, X86_ps=None, y_ps=None, user_ps=None, tag=""):
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(n_members):
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_member(X_raw, user, k, e)))
            labs.append(y)
        if X86_ps is not None:
            feats.append(X86_ps); labs.append(y_ps)
            for k in range(K_AUG_PSEUDO):
                feats.append(build_features(aug_member(Xte_raw_ps, user_ps, k, e,
                                                       seed_base=PSEUDO_SEED)))
                labs.append(y_ps)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        p = dict(full); p["random_state"] = L.SEED + seed_offset + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  {tag} member {e}/{n_members-1} done  (rows={len(Xtr)})")
    return proba / n_members


def full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior):
    print(f"Pass 1 (N={N1}, original only) ...")
    pr1 = run_pass(X_raw, X86, y, user, X86_te, N1, 0, tag="P1")

    mask = pr1.max(axis=1) > THRESH
    y_ps = pr1.argmax(axis=1)[mask]
    X86_ps = X86_te[mask]
    Xte_raw_ps = Xte_raw[mask]
    user_ps = user_te[mask]
    print(f"\n  pseudo-labeled {mask.sum()}/{len(X86_te)} ({mask.mean()*100:.1f}%), "
          f"mean conf={pr1.max(1)[mask].mean():.4f}")
    print(f"  class dist: {dict(sorted(Counter(y_ps.tolist()).items()))}")

    print(f"\nPass 2 (N={N2}, original + {mask.sum()} pseudo, K_aug_pseudo={K_AUG_PSEUDO}) ...")
    pr2 = run_pass(X_raw, X86, y, user, X86_te, N2, 500,
                   Xte_raw_ps=Xte_raw_ps, X86_ps=X86_ps, y_ps=y_ps, user_ps=user_ps, tag="P2")
    return L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    print("=== Pseudo-labeling v4: stronger pass1 + K_pseudo=2 ===\n")
    p1 = full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior)

    print("\nReproducibility check ...")
    p2 = full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior)
    print(f"Reproducible: {np.array_equal(p1, p2)}")
    assert np.array_equal(p1, p2), "NOT reproducible!"

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
