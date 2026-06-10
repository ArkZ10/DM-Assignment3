"""Inter-file DELTA features -- temporal context the HMM doesn't use.

Every model so far treats each file independently: the 86 features capture per-file
statistics, and the HMM captures label-level transitions. Neither uses HOW FEATURES
CHANGE from file to file within a user's sequence.

Delta features: for each file at position t (sorted by file_id within user), append
  delta[t] = X86[t] - X86[t-1]   (first file per user gets zeros)
These are computable at test time without any label leakage (only features used).

Physical intuition for the bottleneck class:
  L2 (slow walking) vs L1 (normal walking): if activity is TRANSITIONING from standing
  (L0) to slow walk (L2), the delta in std channels rises gradually -- different pattern
  than the sharp rise seen in L0->L1 (sudden onset of normal walking).
  The HMM sees only the label boundary; delta features see the feature ramp.

86 feats -> 172 feats (86 raw + 86 delta). Fair 3-partition gate vs 0.7287 ref.
Run: python stepDeltaContext.py
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features


def add_delta(X86, fid, user):
    """Append delta-from-previous-file features; first file per user gets zeros."""
    delta = np.zeros_like(X86)
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        idx_sorted = idx[np.argsort(fid[idx])]
        delta[idx_sorted[1:]] = X86[idx_sorted[1:]] - X86[idx_sorted[:-1]]
    return np.hstack([X86, delta])


def main():
    X_raw, meta = L.D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    groups = user
    X86 = build_features(X_raw)
    Xd = add_delta(X86, fid, user)
    print(f"features: {X86.shape[1]} -> {Xd.shape[1]}  (+{Xd.shape[1]-X86.shape[1]} delta)\n")

    print("Reference: frozen 86-feat LGB+HMM on 3 distinct partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    base_ms = np.array([p[3] for p in parts])
    print(f"  86-feat mean = {base_ms.mean():.4f}  {[round(m,4) for m in base_ms]}\n")

    print("Candidate: 86+delta LGB+HMM (fresh OOF fit, no cache)...")
    new_ms = []
    for s, cv, _lgb_p, _bh in parts:
        probs = L.lgb_oof_probs(Xd, y, groups, cv)
        pred = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        m = f1_score(y, pred, average="macro")
        new_ms.append(m)
        print(f"  partition {s}: macro-F1 = {m:.4f}")
    new_ms = np.array(new_ms)
    print(f"  86+delta mean = {new_ms.mean():.4f}  {[round(m,4) for m in new_ms]}\n")

    print("=" * 60)
    print(f"delta mean: {new_ms.mean()-base_ms.mean():+.4f}   (noise floor {Dl.NOISE})")
    win = Dl.gate(new_ms, base_ms)
    print(f"ROBUST WIN (mean gain > noise AND not worse on any partition): {win}")

    if win:
        print("\n>>> robust win -> writing submission_delta.csv (full refit)")
        Xte_raw, meta_te = L.D.load_split("test")
        fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
        X86_te = build_features(Xte_raw)
        Xd_te = add_delta(X86_te, fid_te, user_te)
        L.write_submission(Xd, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_delta.csv", X_test=Xd_te)
    else:
        print("\n>>> not a robust win.")
        print("    If delta mean is still positive, consider building a delta-feature")
        print("    augmented ensemble (gen_aug_delta_ens.py) for LB testing.")


if __name__ == "__main__":
    main()
