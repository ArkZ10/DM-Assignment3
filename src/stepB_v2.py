"""STEP B v2 -- per-class FB-posterior calibration, re-derived under the FAIR
decoder_lib 3-partition gate (the original stepB_thresholds.py used the STALE
StratifiedGroupKFold partitions -- inflated reference, not directly trustworthy).

Uses the cached 86-feat LGB OOF probs from decoder_lib.get_partitions (same probs
used for the 0.7287 Viterbi reference). For each of the 3 GroupPartition splits,
compute forward-backward posteriors (decode="fb", same s/e/beta/trans/prob_mode as
L.CURRENT). Tune 6 per-class log-multipliers via coordinate ascent on partition 0
(in-sample), then check Dl.gate([fb+mult macro x3], [viterbi macro x3]).

If robust: derive class_mult on the FULL training set's FB posteriors (all data,
no CV) and apply it as a decode-time correction to the pseudo-aug ensemble's test
probabilities (orthogonal to which model produced the probs -- it's a calibration
of the decoder, not the model).

Run: python stepB_v2.py
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features

CFG_FB = dict(s=L.CURRENT["s"], e=L.CURRENT["e"], beta=L.CURRENT["beta"],
               decode="fb", trans=L.CURRENT["trans"], prob_mode=L.CURRENT["prob_mode"])


def macro_with_mult(post, y, logm):
    m = np.exp(logm)
    return f1_score(y, (post * m).argmax(1), average="macro")


def tune_mult(post, y):
    logm = np.zeros(L.N_CLASSES)
    best = macro_with_mult(post, y, logm)
    grid = np.linspace(-1.5, 1.5, 31)
    for _ in range(6):
        improved = False
        for c in range(1, L.N_CLASSES):
            base = logm[c]; bestv, bestf = base, best
            for g in grid:
                logm[c] = g; f = macro_with_mult(post, y, logm)
                if f > bestf:
                    bestf, bestv = f, g
            logm[c] = bestv
            if bestf > best + 1e-9:
                best = bestf; improved = True
        if not improved:
            break
    return logm, best


def main():
    X_raw, meta = L.D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)

    print("Loading 3 fair partitions (cached 86-feat OOF probs)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    hmm_macros = np.array([p[3] for p in parts])
    print(f"  Viterbi (L.CURRENT) reference: mean={hmm_macros.mean():.4f} "
          f"{[round(m,4) for m in hmm_macros]}\n")

    print("Computing FB posteriors per partition...")
    posts = []
    for s, cv, probs, _hmm in parts:
        _, post = L.smooth(probs, y, fid, user, user, cv, return_proba=True, **CFG_FB)
        posts.append(post)

    fb_macros = np.array([f1_score(y, p.argmax(1), average="macro") for p in posts])
    print(f"  FB argmax (no mult): mean={fb_macros.mean():.4f} "
          f"{[round(m,4) for m in fb_macros]}\n")

    print("Tuning class_mult on partition 0 (in-sample)...")
    logm, p0_tuned = tune_mult(posts[0], y)
    mult = np.exp(logm)
    print(f"  partition0 in-sample F1: {fb_macros[0]:.4f} -> {p0_tuned:.4f}")
    print(f"  class_mult: {[round(float(x),3) for x in mult]}\n")

    dec_macros = np.array([macro_with_mult(posts[i], y, logm) for i in range(3)])
    print(f"FB+mult macros (all 3 partitions): {[round(m,4) for m in dec_macros]}")
    print(f"  mean={dec_macros.mean():.4f} vs Viterbi mean={hmm_macros.mean():.4f}  "
          f"delta={dec_macros.mean()-hmm_macros.mean():+.4f}  (noise={Dl.NOISE})")

    win = Dl.gate(dec_macros, hmm_macros)
    print(f"\nROBUST WIN (FB+class_mult vs Viterbi, all 3 partitions): {win}")

    if win:
        print("\nper-class F1 on partitions 1&2 (held-out for tuning), viterbi -> fb+mult:")
        for i in [1, 2]:
            v_pred = L.smooth(parts[i][2], y, fid, user, user, parts[i][1], **L.CURRENT)
            pc_v = f1_score(y, v_pred, average=None, labels=list(range(L.N_CLASSES)))
            pc_m = f1_score(y, (posts[i]*mult).argmax(1), average=None, labels=list(range(L.N_CLASSES)))
            print(f"  partition{i}: " + " ".join(f"L{c}:{pc_v[c]:.3f}->{pc_m[c]:.3f}" for c in range(L.N_CLASSES)))
        print("\n>>> robust win -> save class_mult for application to pseudo-aug ensemble")
        np.save("/root/dm-assignment3/cache/class_mult.npy", mult)
        print(f"saved class_mult to cache/class_mult.npy: {mult}")
    else:
        print("\n>>> not a robust win -> per-class calibration doesn't transfer; skip this lever")


if __name__ == "__main__":
    main()
