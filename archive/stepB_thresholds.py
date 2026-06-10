"""STEP B -- per-class decision-threshold (bias) optimization on smoothed probabilities.

argmax is not macro-F1-optimal under imbalance. We take the smoother's forward-backward
POSTERIORS (probabilities; Viterbi gives only a hard path) and tune 6 per-class
multipliers m_c, predicting argmax_c (post_c * m_c). To avoid overfitting the thresholds:
TUNE on partition 1 only, then APPLY the same m_c to the held-out partitions 2 & 3 and
require it still helps there (robustness gate). No retraining.

Run: python stepB_thresholds.py
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import temporal_lib as L

# emission config = current best (forward-backward to expose posteriors)
CFG = dict(s=L.CURRENT["s"], e=L.CURRENT["e"], beta=L.CURRENT["beta"],
           decode="fb", trans=L.CURRENT["trans"], prob_mode=L.CURRENT["prob_mode"])


def macro_with_mult(post, y, logm):
    m = np.exp(logm)
    return f1_score(y, (post * m).argmax(1), average="macro")


def tune_mult(post, y):
    """Coordinate ascent over log-multipliers (m_0 fixed = 1 for identifiability)."""
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
    X, y, fid, user = L.load_train()
    groups = user
    print("Computing base OOF probabilities for 3 partitions...")
    parts = L.make_partitions(X, y, groups)

    # forward-backward posteriors per partition
    posts = []
    for cv, probs in parts:
        _, post = L.smooth(probs, y, fid, user, groups, cv, return_proba=True, **CFG)
        posts.append(post)

    base_ms = np.array([f1_score(y, p.argmax(1), average="macro") for p in posts])
    print(f"smoothed (argmax, fb) per partition: mean={base_ms.mean():.4f} "
          f"{[round(m,4) for m in base_ms]}\n")

    # TUNE on partition 0, APPLY to 1 & 2 (held-out)
    logm, tuned_p0 = tune_mult(posts[0], y)
    applied = np.array([macro_with_mult(posts[i], y, logm) for i in range(len(posts))])
    print(f"thresholds tuned on partition0 (in-sample F1={tuned_p0:.4f})")
    print(f"applied to all partitions: {[round(m,4) for m in applied]}")
    print(f"multipliers exp(logm): {[round(float(x),3) for x in np.exp(logm)]}")

    # held-out gate: improvement on partitions 1 & 2 (not used for tuning)
    held = applied[1:]; held_base = base_ms[1:]
    held_win = (held.mean() - held_base.mean() > L.NOISE) and bool(np.all(held >= held_base - 1e-9))

    pc_base = f1_score(y, posts[1].argmax(1), average=None, labels=list(range(L.N_CLASSES)))
    pc_thr = f1_score(y, (posts[1] * np.exp(logm)).argmax(1), average=None,
                      labels=list(range(L.N_CLASSES)))
    print("\nper-class F1 on held-out partition1 (argmax -> thresholded):")
    for c in range(L.N_CLASSES):
        print(f"  L{c}: {pc_base[c]:.4f} -> {pc_thr[c]:.4f} ({pc_thr[c]-pc_base[c]:+.4f})")
    print(f"\nHELD-OUT gate (mean +>{L.NOISE} on partitions 1&2 AND not worse on either): {held_win}")

    if held_win:
        print("\n>>> robust win -> writing submission_thresholds.csv")
        L.write_submission(X, y, fid, user, CFG,
                           "/root/dm-assignment3/submission_thresholds.csv",
                           class_mult=np.exp(logm))
    else:
        print("\n>>> not a robust win -> keep submission_temporal.csv.")


if __name__ == "__main__":
    main()
