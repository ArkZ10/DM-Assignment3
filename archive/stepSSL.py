"""Idea 1, step 2 -- LightGBM+HMM on SSL embeddings, fair 3-partition gate.

Uses the cached transductive SSL embeddings (from pretrain_ssl.py). Tests two feature
sets vs the base HMM (0.7287): SSL embeddings alone, and 86 features + SSL embeddings.
Gate: mean > base HMM mean + 0.0024 AND not worse on any partition. Writes
submission_ssl.csv only if the best variant passes.

Run AFTER pretrain_ssl.py:  python stepSSL.py
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import temporal_lib as L
import decoder_lib as Dl

CACHE = "/root/dm-assignment3/cache"


def main():
    X, y, fid, user = L.load_train()
    groups = user
    etr = np.load(os.path.join(CACHE, "ssl_emb_train.npz"))["e"]
    assert len(etr) == len(X), "embedding/file count mismatch -- rerun pretrain_ssl.py"
    print(f"SSL embeddings: {etr.shape}; base features: {X.shape}")

    feature_sets = {
        "SSL emb only": etr,
        "86 + SSL emb": np.hstack([X, etr]),
    }

    parts = Dl.get_partitions(X, y, fid, user)
    base_hmm = [p[3] for p in parts]
    print(f"base HMM ref (3 distinct partitions): {[round(h,4) for h in base_hmm]} "
          f"mean={np.mean(base_hmm):.4f}\n")

    results = {}
    for name, Xf in feature_sets.items():
        hmm_macros, p0 = [], None
        for i, (s, cv, _bp, _bh) in enumerate(parts):
            probs = L.lgb_oof_probs(Xf, y, groups, cv)
            sm = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
            hmm_macros.append(f1_score(y, sm, average="macro"))
            if i == 0:
                p0 = sm
        results[name] = (hmm_macros, p0)
        print(f"{name}: per-partition+HMM {[round(m,4) for m in hmm_macros]} "
              f"mean={np.mean(hmm_macros):.4f}")

    print("\n" + "=" * 78)
    hdr = f"{'variant':24} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate"
    print(hdr); print("-" * len(hdr))
    print(f"{'base 86 + HMM (ref)':24} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} |  ref")
    best = (None, -1, None)
    for name, (ms, p0) in results.items():
        g = Dl.gate(ms, base_hmm)
        print(f"{name+' + HMM':24} | {np.mean(ms):7.4f} | "
              f"{ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} | {g}")
        pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
        cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
        p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
        r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
        print(f"  per-class: " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
        print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]} (base ~78)")
        if np.mean(ms) > best[1]:
            best = (name, np.mean(ms), (ms, Xf if False else name))

    bname, bmean, _ = best
    bms = results[bname][0]
    if Dl.gate(bms, base_hmm):
        print(f"\n>>> robust win ({bname}) -> writing submission_ssl.csv")
        Xf = feature_sets[bname]
        ete = np.load(os.path.join(CACHE, "ssl_emb_test.npz"))["e"]
        Xtest_raw, _ = L.D.load_split("test")
        Xt = L.build_features(Xtest_raw)
        Xtest_f = ete if bname == "SSL emb only" else np.hstack([Xt, ete])
        L.write_submission(Xf, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_ssl.csv", X_test=Xtest_f)
    else:
        print(f"\n>>> no robust win -> keep submission_temporal.csv (0.7867).")


if __name__ == "__main__":
    main()
