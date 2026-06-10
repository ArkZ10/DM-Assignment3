"""Idea D -- transductive self-training with HMM-smoothed pseudo-labels.

Train the augmented LGB on train users, HMM-smooth predictions on the UNLABELED target
users, keep high-confidence files as pseudo-labels, retrain on train + pseudo, predict
target. This injects the target (test) users' feature distribution into training -- the
same cross-subject axis that augmentation exploited (Springer 2025: cross-subject HAR via
self-training + SSL).

Honest gate (no leakage): in OOF we pseudo-label the held-out VAL users and then predict
them -- exactly mirroring the test-time procedure (true val labels used ONLY for scoring,
never for training, same as the LB). Fair 3-partition gate vs base 0.7287 / aug ~0.734.

Run: python stepSelfTrain.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from lgbm_cv import INTERNAL_VAL_FRAC
from step2_familyB import build_features
from stepAugTTA import TUNED, K_AUG, aug_train          # deterministic-uid augmentation

TAU = 0.85                                              # pseudo-label confidence threshold
OUT = "/root/dm-assignment3/submission_selftrain.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
CURRENT_AUG_MEAN = 0.7341


def train_aug(X_raw, X86, y, user, tr, extra_X=None, extra_y=None, full=False):
    if full:
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_train(X_raw, user, k))); labs.append(y)
        if extra_X is not None:
            feats.append(extra_X); labs.append(extra_y)
        Xtr, ytr = np.vstack(feats), np.concatenate(labs)
        p = dict(TUNED); p["n_estimators"] = 463
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        return m
    sub_tr, sub_es = train_test_split(tr, test_size=INTERNAL_VAL_FRAC,
                                      stratify=y[tr], random_state=L.SEED)
    feats, labs = [X86[sub_tr]], [y[sub_tr]]
    for k in range(K_AUG):
        feats.append(build_features(aug_train(X_raw[sub_tr], user[sub_tr], k))); labs.append(y[sub_tr])
    if extra_X is not None:
        feats.append(extra_X); labs.append(extra_y)
    Xtr, ytr = np.vstack(feats), np.concatenate(labs)
    m = lgb.LGBMClassifier(**TUNED)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr),
          eval_set=[(X86[sub_es], y[sub_es])], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    parts = Dl.get_partitions(X86, y, fid, user)
    base_hmm = [p[3] for p in parts]
    print(f"base ref mean={np.mean(base_hmm):.4f}; current aug ~{CURRENT_AUG_MEAN}; "
          f"self-train tau={TAU}\n")

    aug_only, selftrain, p0 = [], [], None
    for s, cv, _lp, _bh in parts:
        oof0 = np.full(len(y), -1); oof1 = np.full(len(y), -1)
        kept_total = 0
        for tr, va in cv.split(X86, y, user):
            T, prior = L.estimate_transition(y[tr], fid[tr], user[tr])
            m0 = train_aug(X_raw, X86, y, user, tr)
            emis0 = m0.predict_proba(X86[va])
            pred0 = L.decode_test(emis0, fid[va], user[va], T, prior, **L.CURRENT)
            oof0[va] = pred0
            # pseudo-label confident val files (HMM-smoothed labels)
            conf = emis0.max(1) > TAU
            kept_total += int(conf.sum())
            m1 = train_aug(X_raw, X86, y, user, tr,
                           extra_X=X86[va][conf], extra_y=pred0[conf])
            emis1 = m1.predict_proba(X86[va])
            oof1[va] = L.decode_test(emis1, fid[va], user[va], T, prior, **L.CURRENT)
        aug_only.append(f1_score(y, oof0, average="macro"))
        selftrain.append(f1_score(y, oof1, average="macro"))
        if p0 is None:
            p0 = oof1
        print(f"  partition {s}: aug-only+HMM={aug_only[-1]:.4f}  self-train+HMM={selftrain[-1]:.4f}"
              f"  (avg kept/fold {kept_total//5})")

    print("\n" + "=" * 70)
    print(f"{'variant':26} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate")
    print(f"{'base LGB+HMM (ref)':26} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} | ref")
    for name, ms in [("aug-only + HMM", aug_only), ("aug + self-train + HMM", selftrain)]:
        print(f"{name:26} | {np.mean(ms):7.4f} | {ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} "
              f"| {Dl.gate(ms, base_hmm)}")

    pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
    print("\nself-train+HMM per-class (part0): " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]}")
    print(f"\nself-train delta vs aug-only (mean): {np.mean(selftrain)-np.mean(aug_only):+.4f}")
    if Dl.gate(selftrain, base_hmm) and np.mean(selftrain) > CURRENT_AUG_MEAN + Dl.NOISE:
        print(f">>> beats current aug by > noise -> writing {OUT}")
        write_submission(X_raw, X86, y, fid, user)
    else:
        print(">>> within noise of current aug; keep submission_aug.csv (0.7904) unless you want LB test.")


def write_submission(X_raw, X86, y, fid, user):
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    def fit_pred():
        m0 = train_aug(X_raw, X86, y, user, None, full=True)
        emis0 = m0.predict_proba(X86_te)
        pred0 = L.decode_test(emis0, fid_te, user_te, T, prior, **L.CURRENT)
        conf = emis0.max(1) > TAU
        m1 = train_aug(X_raw, X86, y, user, None, extra_X=X86_te[conf],
                       extra_y=pred0[conf], full=True)
        emis1 = m1.predict_proba(X86_te)
        return L.decode_test(emis1, fid_te, user_te, T, prior, **L.CURRENT)

    p1 = fit_pred(); p2 = fit_pred()
    assert np.array_equal(p1, p2), "not reproducible!"
    sample = pd.read_csv(SUB_TEMPLATE)
    pbi = dict(zip(fid_te.tolist(), p1.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print(f"Reproducible. Label dist: {dict(sorted(Counter(out.tolist()).items()))}\nWrote {OUT}")


if __name__ == "__main__":
    main()
