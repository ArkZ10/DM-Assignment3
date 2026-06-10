"""Refinement of temporal smoothing: scaled-LIKELIHOOD emissions.

s=0.5 Viterbi won (+0.0073) but HURT L2 (-0.070): L2's self-transition is only 0.46
(it doesn't form runs), so the sticky L1 self-loop (0.91) absorbs isolated L2 blips.
Root cause: we used the classifier POSTERIOR P(c|x) as the HMM emission and multiplied
by the transition prior -> the class prior is double-counted, biasing toward common
(sticky) classes. The principled fix is the scaled LIKELIHOOD  P(x|c) ∝ P(c|x)/P(c)^beta,
which up-weights rare classes (L2) in the emission and should protect them under
smoothing. We sweep beta (prior correction) x s (smoothing strength).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import f1_score

import har_data as D
from har_cv import N_CLASSES, make_cv
from lgbm_cv import BASE_PARAMS, lgbm_oof_proba
from step2_familyB import build_features
from step13_temporal import estimate_transition, viterbi, EPS

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
X = build_features(X_raw)
groups = user


def smooth(oof_p, s, beta):
    cv = make_cv(stratified=True)
    dummy = np.zeros((len(y), 1))
    out = np.full(len(y), -1, dtype=int)
    for tr, va in cv.split(dummy, y, groups):
        T, prior = estimate_transition(y[tr], fid[tr], user[tr])
        log_T = s * np.log(T + EPS)
        log_prior = s * np.log(prior + EPS)
        prior_corr = beta * np.log(prior + EPS)            # scaled-likelihood correction
        for u in np.unique(user[va]):
            idx = va[user[va] == u]; idx = idx[np.argsort(fid[idx])]
            em = np.log(oof_p[idx] + EPS) - prior_corr     # P(c|x)/P(c)^beta
            out[idx] = viterbi(em, log_T, log_prior)
    return out


def main():
    print("Refinement: scaled-likelihood emissions (beta) x smoothing strength (s)\n")
    oof_p = lgbm_oof_proba(X, y, groups, params=TUNED)
    print(f"base argmax macro: {f1_score(y, oof_p.argmax(1), average='macro'):.4f}\n")

    betas = [0.0, 0.5, 1.0]
    ss = [0.3, 0.5, 0.7, 1.0]
    print(f"{'beta':>5} {'s':>5} | {'macro':>7} | {'L2':>6} {'L5':>6} | L0     L1     L3     L4")
    best = (-1, None, None)
    for beta in betas:
        for s in ss:
            pred = smooth(oof_p, s, beta)
            m = f1_score(y, pred, average="macro")
            pc = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)))
            if m > best[0]:
                best = (m, (beta, s), pc)
            print(f"{beta:>5} {s:>5} | {m:>7.4f} | {pc[2]:.4f} {pc[5]:.4f} | "
                  f"{pc[0]:.4f} {pc[1]:.4f} {pc[3]:.4f} {pc[4]:.4f}")
    bm, (bb, bs), bpc = best
    print(f"\nBest: beta={bb}, s={bs} -> macro {bm:.4f}  (vs base 0.7236, posterior-s0.5 0.7309)")
    print(f"  per-class: " + " ".join(f"L{c}={bpc[c]:.4f}" for c in range(N_CLASSES)))


if __name__ == "__main__":
    main()
