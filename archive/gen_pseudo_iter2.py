"""Iterative pseudo-labeling round 2 -- build on the 0.8044 success.

Round 1 (gen_pseudolabel_ens.py) gave LB 0.8044 (+0.0086 over 0.7958).
The pass-2 model is better calibrated -> its pseudo-labels are higher quality.
Round 2 strategy:
  Pass 1 : same proven 5-member aug ensemble on original data (identical to round 1)
           -> pr1. Same seeds -> same pseudo-labels as before (reproducible bridge).
  Pass 2 : 7-member aug ensemble on original + pseudo1 (THRESH=0.82, same as round 1)
           -> pr2. Better model than round 1 pass-2 (more members).
  Pass 3 : 8-member aug ensemble on original + pseudo2 (THRESH=0.78, lower -> more data)
           -> pr3. Final predictions. HMM decode -> submission.

Lower threshold in pass 3 is justified: pr2 is better calibrated than pr1 was, so
files that were borderline at 0.82 are now more confidently predicted -> safe to include.

Writes submission_pseudo2.csv. Fully reproducible (all seeds deterministic).
Run: python gen_pseudo_iter2.py
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

THRESH1 = 0.82     # pass-1 threshold (same as round 1 -- deterministic bridge)
THRESH2 = 0.78     # pass-2 threshold (lower: better model -> more reliable borderline preds)
N1, N2, N3 = 5, 7, 8   # members per pass (scale up progressively)
OUT = "/root/dm-assignment3/submission_pseudo2.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (L.SEED * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def build_aug_train(X_raw, X86, y, user, member):
    feats, labs = [X86], [y]
    for k in range(K_AUG):
        feats.append(build_features(aug_member(X_raw, user, k, member)))
        labs.append(y)
    return np.vstack(feats), np.concatenate(labs)


def run_ensemble(X_raw, X86, y, user, X86_te, n_members, seed_offset,
                 X86_extra=None, y_extra=None):
    """Train n_members aug LGB ensemble; optionally append extra (pseudo) data."""
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(n_members):
        Xtr, ytr = build_aug_train(X_raw, X86, y, user, e)
        if X86_extra is not None:
            Xtr = np.vstack([Xtr, X86_extra])
            ytr = np.concatenate([ytr, y_extra])
        p = dict(full); p["random_state"] = L.SEED + seed_offset + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  [seed+{seed_offset}] member {e}/{n_members-1} done")
    return proba / n_members


def pseudo_from(probs, X86_te, thresh, label=""):
    mask = probs.max(axis=1) > thresh
    pseudo_X = X86_te[mask]
    pseudo_y = probs.argmax(axis=1)[mask]
    print(f"  {label}pseudo-labeled {mask.sum()}/{len(X86_te)} "
          f"({mask.mean()*100:.1f}%, thresh={thresh:.2f}), "
          f"mean conf={probs.max(1)[mask].mean():.4f}")
    print(f"  class dist: {dict(sorted(Counter(pseudo_y.tolist()).items()))}")
    return pseudo_X, pseudo_y


def full_run(X_raw, X86, y, user, fid, X86_te, fid_te, user_te, T, prior):
    print(f"Pass 1 ({N1} members, original data)...")
    pr1 = run_ensemble(X_raw, X86, y, user, X86_te, N1, seed_offset=0)
    pX1, pY1 = pseudo_from(pr1, X86_te, THRESH1, "pass1 ")

    print(f"\nPass 2 ({N2} members, original + pseudo1)...")
    pr2 = run_ensemble(X_raw, X86, y, user, X86_te, N2, seed_offset=500,
                       X86_extra=pX1, y_extra=pY1)
    pX2, pY2 = pseudo_from(pr2, X86_te, THRESH2, "pass2 ")

    print(f"\nPass 3 ({N3} members, original + pseudo2)...")
    pr3 = run_ensemble(X_raw, X86, y, user, X86_te, N3, seed_offset=1000,
                       X86_extra=pX2, y_extra=pY2)
    return pr3


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    print("=== Iterative pseudo-labeling round 2 ===\n")
    pr = full_run(X_raw, X86, y, user, fid, X86_te, fid_te, user_te, T, prior)
    p1 = L.decode_test(pr, fid_te, user_te, T, prior, **L.CURRENT)

    print("\nReproducibility check (re-running all 3 passes)...")
    pr_b = full_run(X_raw, X86, y, user, fid, X86_te, fid_te, user_te, T, prior)
    p2 = L.decode_test(pr_b, fid_te, user_te, T, prior, **L.CURRENT)
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
