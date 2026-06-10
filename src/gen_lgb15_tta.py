"""LGB-only ensemble (15 members) + TTA -- build on the proven 0.7958 lever.

Mega ensemble (8 LGB + 3 XGB + 3 CAT) worsened because XGB/CAT diluted LGB signal.
This scales the proven approach:
  - 15 LGB members, diverse seeds + augmentation strengths (rot20/rot25/rot30)
  - TTA: each test file averaged over M=4 augmented views + original per member
  - Total: 15 * 5 = 75 probability estimates per test file before HMM

Writes submission_lgb15_tta.csv; reproducibility-checked.
Run: python gen_lgb15_tta.py
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
from stepAugTTA import TUNED, K_AUG, transform, uid

N_MEMBERS = 15
M_TTA = 4                # test-time views per member (+ original = 5 total)
STRENGTHS = [(20.0, 0.10, 0.20), (25.0, 0.12, 0.25), (30.0, 0.15, 0.30)]
OUT = "/root/dm-assignment3/submission_lgb15_tta.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def rot_mat(rng, deg):
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-9)
    a = np.deg2rad(rng.uniform(-deg, deg))
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def aug_train_member(X_raw, user, k, member, deg, sm, ss):
    """Per-user seeded augmentation for training — distinct per (member, k)."""
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (L.SEED * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), deg, sm, ss)
    return out


def tta_proba(model, Xte_raw, member, deg, sm, ss):
    """Average probabilities over original + M TTA views of test set."""
    probs = model.predict_proba(build_features(Xte_raw))
    for v in range(M_TTA):
        seed = (L.SEED * 200003 + member * 333331 + v * 7919) % (2**32)
        rng = np.random.RandomState(seed)
        R = rot_mat(rng, deg)
        a = rng.uniform(1 - sm, 1 + sm)
        b = rng.uniform(1 - ss, 1 + ss)
        mean = (Xte_raw[..., :3] @ R.T) * a
        std = np.abs(Xte_raw[..., 3:] * b)
        Xv = np.concatenate([mean, std], axis=-1)
        probs += model.predict_proba(build_features(Xv))
    return probs / (M_TTA + 1)


def ensemble_proba(X_raw, X86, y, user, Xte_raw):
    lgb_p = dict(TUNED); lgb_p["n_estimators"] = 463
    proba = np.zeros((len(Xte_raw), L.N_CLASSES))
    for e in range(N_MEMBERS):
        deg, sm, ss = STRENGTHS[e % len(STRENGTHS)]
        # build augmented training set
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_train_member(X_raw, user, k, e, deg, sm, ss)))
            labs.append(y)
        Xtr, ytr = np.vstack(feats), np.concatenate(labs)
        p = dict(lgb_p); p["random_state"] = L.SEED + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += tta_proba(m, Xte_raw, e, deg, sm, ss)
        print(f"  member {e:2d} (rot{deg}) done")
    return proba / N_MEMBERS


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    fid = meta["file_id"].to_numpy()
    user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy()
    user_te = meta_te["user"].to_numpy()
    T, prior = L.estimate_transition(y, fid, user)

    print(f"training {N_MEMBERS}-member LGB ensemble + {M_TTA}-view TTA (refit #1)")
    pr1 = ensemble_proba(X_raw, X86, y, user, Xte_raw)
    p1 = L.decode_test(pr1, fid_te, user_te, T, prior, **L.CURRENT)

    print(f"\nrefit #2 (reproducibility check)")
    pr2 = ensemble_proba(X_raw, X86, y, user, Xte_raw)
    p2 = L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)
    print(f"Reproducible: {np.array_equal(p1, p2)}")
    assert np.array_equal(p1, p2), "NOT reproducible — check seeds!"

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
