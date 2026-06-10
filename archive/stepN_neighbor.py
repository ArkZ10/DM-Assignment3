"""Idea 5 experiment: neighbor-feature LightGBM + HMM, fair 3-partition gate.

Compares: base (86 feats) LGB+HMM  vs  neighbor (433 feats) LGB+HMM, on the 3 truly-
distinct partitions. Gate: mean > base HMM mean + 0.0024 AND not worse on any partition.
Reports L2 precision/recall and true1->pred2 (the L2-vs-L1 bottleneck). Writes
submission_neighbor.csv only if the gate passes.

Run: python stepN_neighbor.py
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import temporal_lib as L
import decoder_lib as Dl
from neighbor_features import build_neighbor_features


def main():
    X, y, fid, user = L.load_train()
    groups = user
    Xn = build_neighbor_features(X, fid, user)
    print(f"features: base {X.shape[1]} -> neighbor-augmented {Xn.shape[1]}")

    parts = Dl.get_partitions(X, y, fid, user)          # (seed, cv, base_probs, base_hmm)
    base_hmm = [p[3] for p in parts]
    print(f"base HMM ref (3 distinct partitions): {[round(h,4) for h in base_hmm]} "
          f"mean={np.mean(base_hmm):.4f}\n")

    aug_lgb, aug_hmm, p0_smooth = [], [], None
    for i, (s, cv, _bp, _bh) in enumerate(parts):
        probs = L.lgb_oof_probs(Xn, y, groups, cv)
        aug_lgb.append(f1_score(y, probs.argmax(1), average="macro"))
        sm = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        aug_hmm.append(f1_score(y, sm, average="macro"))
        if i == 0:
            p0_smooth = sm
        print(f"  partition {s}: neighbor LGB(argmax)={aug_lgb[-1]:.4f}  "
              f"neighbor+HMM={aug_hmm[-1]:.4f}  (base+HMM={_bh:.4f})")

    print("\n" + "=" * 78)
    hdr = f"{'variant':28} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7}"
    print(hdr); print("-" * len(hdr))
    print(f"{'base 86 + HMM (ref)':28} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f}")
    print(f"{'neighbor 433 LGB (argmax)':28} | {np.mean(aug_lgb):7.4f} | "
          f"{aug_lgb[0]:7.4f} {aug_lgb[1]:7.4f} {aug_lgb[2]:7.4f}")
    print(f"{'neighbor 433 + HMM':28} | {np.mean(aug_hmm):7.4f} | "
          f"{aug_hmm[0]:7.4f} {aug_hmm[1]:7.4f} {aug_hmm[2]:7.4f}")

    pc = f1_score(y, p0_smooth, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0_smooth, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0_smooth, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0_smooth, labels=[2], average="micro", zero_division=0)
    print("\nneighbor+HMM per-class (partition0): " +
          " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 precision={p2:.4f} recall={r2:.4f} | true1->pred2={cm[1,2]} (base HMM had ~78)")

    g = Dl.gate(aug_hmm, base_hmm)
    print(f"\ndelta mean (neighbor+HMM - base+HMM): {np.mean(aug_hmm)-np.mean(base_hmm):+.4f} "
          f"(noise {Dl.NOISE})")
    print(f"ROBUST WIN (mean +>{Dl.NOISE} AND not worse on any partition): {g}")

    if g:
        print("\n>>> robust win -> writing submission_neighbor.csv")
        Xtest_raw, meta_te = L.D.load_split("test")
        Xt = L.build_features(Xtest_raw)
        Xnt = build_neighbor_features(Xt, meta_te["file_id"].to_numpy(), meta_te["user"].to_numpy())
        L.write_submission(Xn, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_neighbor.csv", X_test=Xnt)
    else:
        print("\n>>> not a robust win -> keep submission_temporal.csv (0.7867).")


if __name__ == "__main__":
    main()
