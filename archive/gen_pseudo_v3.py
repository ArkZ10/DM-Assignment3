"""Pseudo-labeling v3: 3-pass iterative, FIXED threshold 0.82, augmented pseudo data.

Progression so far:
  gen_pseudolabel_ens.py : 2-pass, thresh=0.82, no pseudo-aug     -> LB 0.8044
  gen_pseudo_aug.py      : 2-pass, thresh=0.82, K_pseudo=1, N2=7 -> LB 0.8061
  gen_pseudo_iter2.py    : 3-pass, thresh=0.78 in pass3           -> WORSE (noise)

Key lesson: lower threshold = noisy pseudo-labels = regression.
This version: 3-pass, FIXED thresh=0.82 throughout. The pass-2 model is better
calibrated (LB 0.8061 vs 0.7958 for raw pass-1), so at the SAME threshold it will:
  - pseudo-label more files (more confident about borderline cases)
  - pseudo-label with higher accuracy (better calibration = fewer errors)
Using its pseudo-labels in pass-3 should give a further gain without noise risk.

  Pass 1: N=5,  original only          -> pr1 -> pseudo1 (thresh 0.82)
  Pass 2: N=7,  original + pseudo1_aug -> pr2 -> pseudo2 (thresh 0.82 on better model)
  Pass 3: N=9,  original + pseudo2_aug -> pr3 -> final predictions + HMM

Writes submission_pseudo_v3.csv; reproducible.
Run: python gen_pseudo_v3.py
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
K_AUG_PSEUDO = 1
N1, N2, N3 = 5, 7, 9
PSEUDO_SEED = 777777
OUT = "/root/dm-assignment3/submission_pseudo_v3.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member, seed_base=L.SEED):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (seed_base * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def run_pass(X_raw, X86, y, user, X86_te, n_members, seed_offset,
             Xte_raw_ps=None, X86_ps=None, y_ps=None, user_ps=None, pass_name=""):
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
        print(f"  {pass_name} member {e}/{n_members-1} done")
    return proba / n_members


def get_pseudo(probs, Xte_raw, X86_te, user_te, label):
    mask = probs.max(axis=1) > THRESH
    y_ps = probs.argmax(axis=1)[mask]
    print(f"  {label}: {mask.sum()}/{len(X86_te)} pseudo-labeled "
          f"({mask.mean()*100:.1f}%), mean conf={probs.max(1)[mask].mean():.4f}, "
          f"dist={dict(sorted(Counter(y_ps.tolist()).items()))}")
    return Xte_raw[mask], X86_te[mask], y_ps, user_te[mask]


def full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior):
    print(f"Pass 1 (N={N1}, original only)...")
    pr1 = run_pass(X_raw, X86, y, user, X86_te, N1, 0, pass_name="P1")
    Xr1, X1, y1, u1 = get_pseudo(pr1, Xte_raw, X86_te, user_te, "pseudo1")

    print(f"\nPass 2 (N={N2}, +pseudo1_aug)...")
    pr2 = run_pass(X_raw, X86, y, user, X86_te, N2, 500,
                   Xte_raw_ps=Xr1, X86_ps=X1, y_ps=y1, user_ps=u1, pass_name="P2")
    Xr2, X2, y2, u2 = get_pseudo(pr2, Xte_raw, X86_te, user_te, "pseudo2")

    print(f"\nPass 3 (N={N3}, +pseudo2_aug)...")
    pr3 = run_pass(X_raw, X86, y, user, X86_te, N3, 1000,
                   Xte_raw_ps=Xr2, X86_ps=X2, y_ps=y2, user_ps=u2, pass_name="P3")
    return L.decode_test(pr3, fid_te, user_te, T, prior, **L.CURRENT)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    print("=== Pseudo-labeling v3: 3-pass fixed threshold ===\n")
    p1 = full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior)

    print("\nReproducibility check...")
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
