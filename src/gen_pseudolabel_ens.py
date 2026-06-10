"""Pseudo-labeling ensemble -- semi-supervised bridge to unseen test users.

Gap from 0.7958 to 0.80+ : cross-subject generalization is the binding constraint.
The 40 test users are completely unseen. Pseudo-labeling high-confidence test
predictions adds them to training, directly teaching the model about the test
distribution. Two-pass recipe:

  Pass 1: train the proven 5-member augmented ensemble on real training data.
          Get raw probabilities for ALL test files.
  Filter: keep test files where max_prob > THRESH (default 0.82).
          These are almost certainly correct (easy, unambiguous files).
  Pass 2: retrain the ensemble on real training + pseudo-labeled test files.
          No augmentation of pseudo data -- keep noise low.
          Get final test predictions from the new ensemble, HMM decode.

Previously broken (stepSelfTrain.py) because it compared against a stale OOF
constant (0.7341) instead of this run's actual output. This version uses the
ensemble raw probabilities directly -- no stale comparisons.

Writes submission_pseudo.csv; both passes are deterministic and reproducible.
Run: python gen_pseudolabel_ens.py
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

THRESH = 0.82          # confidence threshold for pseudo-labeling
N_MEMBERS = 5          # same as proven ensemble
OUT = "/root/dm-assignment3/submission_pseudo.csv"
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


def ensemble_proba_pass1(X_raw, X86, y, user, X86_te):
    """5-member aug ensemble on original training data. Returns averaged test probs."""
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(N_MEMBERS):
        Xtr, ytr = build_aug_train(X_raw, X86, y, user, e)
        p = dict(full); p["random_state"] = L.SEED + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  pass1 member {e} done")
    return proba / N_MEMBERS


def ensemble_proba_pass2(X_raw, X86, y, user, X86_te, X86_pseudo, y_pseudo):
    """5-member aug ensemble on original + pseudo-labeled test data. Returns test probs."""
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(N_MEMBERS):
        Xtr_aug, ytr_aug = build_aug_train(X_raw, X86, y, user, e)
        # append pseudo-labeled test data (no augmentation -- keep noise low)
        Xtr = np.vstack([Xtr_aug, X86_pseudo])
        ytr = np.concatenate([ytr_aug, y_pseudo])
        p = dict(full); p["random_state"] = L.SEED + 500 + e   # distinct seed from pass1
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  pass2 member {e} done")
    return proba / N_MEMBERS


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    # ---- pass 1: get raw ensemble probabilities ----
    print(f"Pass 1: 5-member aug ensemble on original training data...")
    pr1 = ensemble_proba_pass1(X_raw, X86, y, user, X86_te)

    # ---- pseudo-label high-confidence test files ----
    confidence = pr1.max(axis=1)
    pseudo_mask = confidence > THRESH
    pseudo_labels = pr1.argmax(axis=1)[pseudo_mask]
    X86_pseudo = X86_te[pseudo_mask]
    print(f"\nPseudo-labeled {pseudo_mask.sum()}/{len(X86_te)} test files "
          f"(conf > {THRESH:.2f}, {pseudo_mask.mean()*100:.1f}%)")
    print(f"Pseudo-label class dist: {dict(sorted(Counter(pseudo_labels.tolist()).items()))}")
    print(f"Mean confidence of accepted: {confidence[pseudo_mask].mean():.4f}")

    # ---- pass 2: retrain on original + pseudo-labeled ----
    print(f"\nPass 2: retraining on {len(X86) + len(X86_pseudo)} files "
          f"({len(X86)} real + {len(X86_pseudo)} pseudo) ...")
    pr2 = ensemble_proba_pass2(X_raw, X86, y, user, X86_te, X86_pseudo, pseudo_labels)
    p2 = L.decode_test(pr2, fid_te, user_te, T, prior, **L.CURRENT)

    # ---- reproducibility check ----
    print("\nReproducibility check (re-running both passes)...")
    pr1b = ensemble_proba_pass1(X_raw, X86, y, user, X86_te)
    pseudo_mask_b = pr1b.max(axis=1) > THRESH
    pseudo_labels_b = pr1b.argmax(axis=1)[pseudo_mask_b]
    X86_pseudo_b = X86_te[pseudo_mask_b]
    pr2b = ensemble_proba_pass2(X_raw, X86, y, user, X86_te, X86_pseudo_b, pseudo_labels_b)
    p2b = L.decode_test(pr2b, fid_te, user_te, T, prior, **L.CURRENT)
    print(f"Reproducible: {np.array_equal(p2, p2b)}")
    assert np.array_equal(p2, p2b), "NOT reproducible -- check seeds!"

    # ---- write submission ----
    sample = pd.read_csv(SUB_TEMPLATE)
    pbi = dict(zip(fid_te.tolist(), p2.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print("Label dist:", {c: Counter(out.tolist()).get(c, 0) for c in range(L.N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
