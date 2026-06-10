"""Robustness / anti-overfit check for the temporal smoother.

The submitted config (Viterbi, empirical 6x6 transition, beta=0.5, s=0.7) was picked
from ONE 15-cell grid on ONE fold partition -> mild selection-optimism risk. Here we:
  1. REPEAT the CV over several fold partitions (StratifiedGroupKFold shuffle seeds) and
     report each config's mean +/- std OOF macro-F1 -> selects what GENERALIZES, not what
     wins one partition.
  2. Compare lower-overfit-risk variants:
       - decode: Viterbi (hard path) vs Forward-Backward (soft posterior marginals)
       - transition: full empirical 6x6 (30 free off-diags) vs "sticky" (per-class
         diagonal kept, off-diagonals uniformized -> 0 free off-diag params, more robust)
We prefer the config with the best MEAN that is also stable (low std), leaning
conservative when means tie.

Run: python step14_robust.py
"""
from __future__ import annotations
import json
import numpy as np
import lightgbm as lgb
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, INTERNAL_VAL_FRAC
from step2_familyB import build_features
from step13_temporal import estimate_transition, viterbi, EPS

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
PARTITION_SEEDS = [42, 7, 123]
SUBMITTED = ("viterbi", "emp", 0.5, 0.7)   # current submission config


def oof_proba_with_cv(X, y, groups, cv):
    oof = np.zeros((len(y), N_CLASSES))
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(
            X[tr], y[tr], test_size=INTERNAL_VAL_FRAC, stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**TUNED)
        m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X[va])
    return oof


def sticky_transition(T):
    K = len(T); diag = np.diag(T).copy()
    Ts = np.empty_like(T)
    for i in range(K):
        Ts[i, :] = (1 - diag[i]) / (K - 1)
        Ts[i, i] = diag[i]
    return Ts


def forward_backward(log_em, log_T, log_prior):
    n, K = log_em.shape
    la = np.empty((n, K)); la[0] = log_prior + log_em[0]
    for t in range(1, n):
        la[t] = log_em[t] + logsumexp(la[t - 1][:, None] + log_T, axis=0)
    lb = np.zeros((n, K))
    for t in range(n - 2, -1, -1):
        lb[t] = logsumexp(log_T + (log_em[t + 1] + lb[t + 1])[None, :], axis=1)
    return (la + lb).argmax(1)


def smooth(oof_p, y, fid, user, groups, cv, decode, trans, beta, s):
    out = np.full(len(y), -1, dtype=int)
    dummy = np.zeros((len(y), 1))
    for tr, va in cv.split(dummy, y, groups):
        T, prior = estimate_transition(y[tr], fid[tr], user[tr])
        if trans == "sticky":
            T = sticky_transition(T)
        log_T = s * np.log(T + EPS)
        log_prior = s * np.log(prior + EPS)
        prior_corr = beta * np.log(prior + EPS)
        decoder = viterbi if decode == "viterbi" else forward_backward
        for u in np.unique(user[va]):
            idx = va[user[va] == u]; idx = idx[np.argsort(fid[idx])]
            em = np.log(oof_p[idx] + EPS) - prior_corr
            out[idx] = decoder(em, log_T, log_prior)
    return out


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    groups = user
    X = build_features(X_raw)

    decodes = ["viterbi", "fb"]
    transes = ["emp", "sticky"]
    betas = [0.3, 0.5]
    ss = [0.3, 0.5, 0.7, 1.0]

    # accumulate macro per config across partitions; also track base per partition
    acc = {}
    base_macros = []
    for ps in PARTITION_SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=ps)
        oof_p = oof_proba_with_cv(X, y, groups, cv)
        base_macros.append(f1_score(y, oof_p.argmax(1), average="macro"))
        for decode in decodes:
            for trans in transes:
                for beta in betas:
                    for s in ss:
                        pred = smooth(oof_p, y, fid, user, groups, cv, decode, trans, beta, s)
                        m = f1_score(y, pred, average="macro")
                        l2 = f1_score(y, pred, average=None, labels=[2])[0]
                        acc.setdefault((decode, trans, beta, s), []).append((m, l2))

    print(f"Base OOF macro across {len(PARTITION_SEEDS)} partitions: "
          f"mean={np.mean(base_macros):.4f} std={np.std(base_macros):.4f} "
          f"{[round(b,4) for b in base_macros]}\n")

    rows = []
    for cfg, vals in acc.items():
        ms = np.array([v[0] for v in vals]); l2s = np.array([v[1] for v in vals])
        rows.append((cfg, ms.mean(), ms.std(), l2s.mean()))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'decode':8} {'trans':7} {'beta':>4} {'s':>4} | {'mean':>7} {'std':>6} {'meanL2':>7}")
    print("-" * 52)
    for cfg, mean, std, l2 in rows[:12]:
        d, t, b, s = cfg
        mark = "  <- SUBMITTED" if cfg == SUBMITTED else ""
        print(f"{d:8} {t:7} {b:>4} {s:>4} | {mean:>7.4f} {std:>6.4f} {l2:>7.4f}{mark}")

    sub = next((r for r in rows if r[0] == SUBMITTED), None)
    best = rows[0]
    print(f"\nSubmitted config {SUBMITTED}: mean={sub[1]:.4f} std={sub[2]:.4f}")
    print(f"Best mean config {best[0]}: mean={best[1]:.4f} std={best[2]:.4f}")
    print(f"Delta best-vs-submitted: {best[1]-sub[1]:+.4f} "
          f"(noise floor on a single partition ~0.0024; across-partition std shown above)")


if __name__ == "__main__":
    main()
