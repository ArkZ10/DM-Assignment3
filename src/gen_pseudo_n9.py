"""Trial 2 of 2: minimal, isolated scale-up of the proven 0.8061 recipe.

gen_pseudo_aug.py : N_PASS1=5, N_PASS2=7, K_AUG_PSEUDO=1, thresh=0.82 -> LB 0.8061
gen_pseudo_v4.py  : N_PASS1=10, N_PASS2=10, K_AUG_PSEUDO=2 (2 vars changed)-> LB 0.799

v4 changed THREE things at once (pass-1 strength, pass-2 size, K_pseudo) and
regressed -- can't tell which hurt. This trial isolates ONE variable: N_PASS2
7 -> 9 (modest +2), keeping pass-1 at 5 (same pseudo-labels, same bridge),
K_AUG_PSEUDO=1, thresh=0.82, Viterbi decode -- everything else byte-identical
to gen_pseudo_aug.py.

Writes submission_pseudo_n9.csv; reproducible.
Run: python gen_pseudo_n9.py
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
N_PASS1 = 5
N_PASS2 = 9        # only change vs gen_pseudo_aug.py (was 7)
PSEUDO_SEED = 777777
OUT = "/root/dm-assignment3/submission_pseudo_n9.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member, seed_base=L.SEED):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (seed_base * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def run_ensemble(X_raw, X86, y, user, X86_te, n_members, seed_offset,
                 Xte_raw_pseudo=None, X86_pseudo=None, y_pseudo=None, user_pseudo=None):
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(n_members):
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_member(X_raw, user, k, e)))
            labs.append(y)
        if X86_pseudo is not None:
            feats.append(X86_pseudo); labs.append(y_pseudo)
            for k in range(K_AUG_PSEUDO):
                feats.append(build_features(aug_member(Xte_raw_pseudo, user_pseudo,
                                                       k, e, seed_base=PSEUDO_SEED)))
                labs.append(y_pseudo)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        p = dict(full); p["random_state"] = L.SEED + seed_offset + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  [offset={seed_offset}] member {e}/{n_members-1} done")
    return proba / n_members


def full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior):
    print(f"Pass 1 ({N_PASS1} members, original data only)...")
    pr1 = run_ensemble(X_raw, X86, y, user, X86_te, N_PASS1, seed_offset=0)

    mask = pr1.max(axis=1) > THRESH
    y_pseudo = pr1.argmax(axis=1)[mask]
    X86_pseudo = X86_te[mask]
    Xte_raw_pseudo = Xte_raw[mask]
    user_pseudo = user_te[mask]
    print(f"\nPseudo-labeled {mask.sum()}/{len(X86_te)} "
          f"({mask.mean()*100:.1f}%, thresh={THRESH}), "
          f"mean conf={pr1.max(1)[mask].mean():.4f}")
    print(f"Class dist: {dict(sorted(Counter(y_pseudo.tolist()).items()))}")

    rows_real = len(X86) * (1 + K_AUG)
    rows_pseudo = len(X86_pseudo) * (1 + K_AUG_PSEUDO)
    print(f"\nPass 2 ({N_PASS2} members, {rows_real} real + {rows_pseudo} pseudo rows)...")
    pr2 = run_ensemble(X_raw, X86, y, user, X86_te, N_PASS2, seed_offset=500,
                       Xte_raw_pseudo=Xte_raw_pseudo, X86_pseudo=X86_pseudo,
                       y_pseudo=y_pseudo, user_pseudo=user_pseudo)
    return L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    print("=== Trial 2: N_PASS2 7 -> 9 (isolated +2 increment) ===\n")
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
