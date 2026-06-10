"""Per-user feature normalization -- directly attacks the cross-subject shift.

All prior diversity attempts (more members, XGB/CAT, spectral features) came back flat or
worse. They add variations of the SAME information. The binding constraint is
cross-subject generalization (60 train -> 40 unseen test users). Per-user normalization
removes user-level distribution shift without any adversarial training:

  For each OOF fold: train-users normalize by their own train-fold stats,
                     val-users normalize by their own val-fold stats (mimics test time).
  For test:          each test user normalizes by their own test files.

After normalization the model sees only activity signal (within-user relative patterns),
not subject signal (absolute offsets from device orientation, carrying style, body size).
This is a standard HAR cross-subject trick that costs zero extra compute.

Three-partition gate vs frozen 86-feat LGB+HMM (0.7287 ref). CPU, deterministic.
Run: python stepUserNorm.py
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

import temporal_lib as L
import decoder_lib as Dl
from lgbm_cv import INTERNAL_VAL_FRAC
from step2_familyB import build_features


def user_normalize(X, user):
    """Z-score each user's rows by that user's own mean/std. Leak-free when called per-fold."""
    Xn = X.copy().astype(float)
    for u in np.unique(user):
        m = user == u
        mu = X[m].mean(0)
        sd = X[m].std(0) + 1e-8
        Xn[m] = (X[m] - mu) / sd
    return Xn


def oof_normalized(X86, y, user, cv):
    """OOF probs with per-fold, per-user normalization (leak-free)."""
    oof = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X86, y, user):
        Xn_tr = user_normalize(X86[tr], user[tr])
        Xn_va = user_normalize(X86[va], user[va])
        Xi, Xe, yi, ye = train_test_split(Xn_tr, y[tr], test_size=INTERNAL_VAL_FRAC,
                                          stratify=y[tr], random_state=L.SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**L.TUNED)
        m.fit(Xi, yi, sample_weight=sw,
              eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(Xn_va)
    return oof


def main():
    X_raw, meta = L.D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    groups = user
    X86 = build_features(X_raw)

    print("Reference: frozen 86-feat LGB+HMM on 3 distinct partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    base_ms = np.array([p[3] for p in parts])
    print(f"  86-feat mean = {base_ms.mean():.4f}  {[round(m, 4) for m in base_ms]}\n")

    print("Candidate: per-user-normalised 86-feat LGB+HMM (fresh OOF, leak-free)...")
    new_ms = []
    for s, cv, _lgb_p86, _bh in parts:
        probs = oof_normalized(X86, y, user, cv)
        pred = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        m = f1_score(y, pred, average="macro")
        new_ms.append(m)
        print(f"  partition {s}: macro-F1 = {m:.4f}")
    new_ms = np.array(new_ms)
    print(f"  user-norm mean = {new_ms.mean():.4f}  {[round(m, 4) for m in new_ms]}\n")

    print("=" * 60)
    print(f"delta mean: {new_ms.mean() - base_ms.mean():+.4f}   (noise floor {Dl.NOISE})")
    win = Dl.gate(new_ms, base_ms)
    print(f"ROBUST WIN (mean gain > noise AND not worse on any partition): {win}")

    if win:
        print("\n>>> robust win -> writing submission_usernorm.csv (full refit, repro-checked)")
        Xte_raw, meta_te = L.D.load_split("test")
        fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
        X86_te = build_features(Xte_raw)
        X86_tr_n = user_normalize(X86, user)
        X86_te_n = user_normalize(X86_te, user_te)
        L.write_submission(X86_tr_n, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_usernorm.csv",
                           X_test=X86_te_n)
    else:
        print("\n>>> not a robust win -> keep aug ensemble (LB 0.7958).")


if __name__ == "__main__":
    main()
