"""User-deviation features: X86[t] - user_mean -- within-user context, additive.

User normalization (replacing X86 with z-scores) hurt badly (-0.0605) by destroying
absolute scale. This version ADDS the deviation instead of replacing:
  X_dev[t] = X86[t] - mean_over_user_files(X86)
=> 86 + 86 = 172 features. All original information preserved.

Physical intuition for the L2 bottleneck: if a user's typical activity is L1 walking,
an L2 slow-walk file will show NEGATIVE deviation in std channels (lower activity than
their baseline) -- a discriminative signal that absolute features + HMM can't capture
directly but the tree can pick up from the deviation column.

Leak-free: train-users' deviations computed from their own train-fold files;
val-users' deviations from their own val-fold files (mirrors test-time where we compute
each test user's mean from all their test files -- no label access needed).

Fair 3-partition gate vs 86-feat LGB+HMM (0.7287 ref).
Run: python stepUserDev.py
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features


def add_user_dev(X86, user, ref_X=None, ref_user=None):
    """Append X86[t] - user_mean. ref_X/ref_user: source for computing user means
    (train fold at OOF time; same set at test time)."""
    src_X = ref_X if ref_X is not None else X86
    src_user = ref_user if ref_user is not None else user
    dev = np.zeros_like(X86)
    for u in np.unique(user):
        u_src = src_user == u
        u_tgt = user == u
        if u_src.sum() > 0:
            dev[u_tgt] = X86[u_tgt] - src_X[u_src].mean(0)
    return np.hstack([X86, dev])


def oof_with_dev(X86, y, fid, user, cv):
    """OOF probs using 86+user_dev features, leak-free per fold."""
    oof = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X86, y, user):
        # train users: deviate from their own train-fold mean
        Xtr = add_user_dev(X86[tr], user[tr])
        # val users: deviate from their own val-fold mean (mirrors test-time)
        Xva = add_user_dev(X86[va], user[va])
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_sample_weight
        from lgbm_cv import INTERNAL_VAL_FRAC
        Xi, Xe, yi, ye = train_test_split(Xtr, y[tr], test_size=INTERNAL_VAL_FRAC,
                                          stratify=y[tr], random_state=L.SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**L.TUNED)
        m.fit(Xi, yi, sample_weight=sw,
              eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(Xva)
    return oof


def main():
    X_raw, meta = L.D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    print(f"features: {X86.shape[1]} -> {X86.shape[1]*2}  (+{X86.shape[1]} user-deviation)\n")

    print("Reference: frozen 86-feat LGB+HMM on 3 distinct partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    base_ms = np.array([p[3] for p in parts])
    print(f"  86-feat mean = {base_ms.mean():.4f}  {[round(m,4) for m in base_ms]}\n")

    print("Candidate: 86+user_dev LGB+HMM (fresh OOF, leak-free)...")
    new_ms = []
    for s, cv, _lgb_p, _bh in parts:
        probs = oof_with_dev(X86, y, fid, user, cv)
        pred = L.smooth(probs, y, fid, user, user, cv, **L.CURRENT)
        m = f1_score(y, pred, average="macro")
        new_ms.append(m)
        print(f"  partition {s}: macro-F1 = {m:.4f}")
    new_ms = np.array(new_ms)
    print(f"  86+user_dev mean = {new_ms.mean():.4f}  {[round(m,4) for m in new_ms]}\n")

    print("=" * 60)
    print(f"delta mean: {new_ms.mean()-base_ms.mean():+.4f}   (noise floor {Dl.NOISE})")
    win = Dl.gate(new_ms, base_ms)
    print(f"ROBUST WIN: {win}")

    if win:
        print("\n>>> robust win -> can integrate into pseudo-labeling pipeline for LB test")
    else:
        print("\n>>> not a robust win -> user-deviation features don't help standalone")


if __name__ == "__main__":
    main()
