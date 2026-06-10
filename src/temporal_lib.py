"""Shared library for the HMM-smoother optimization steps (A/B/C).

One honest harness: base OOF probabilities are computed ONCE per fold partition (they
don't depend on smoother hyperparameters), then smoothing/thresholding is applied on the
cached probs -> fast sweeps. Robustness = evaluate every config across 3 fold partitions
(StratifiedGroupKFold shuffle seeds) and require consistency, not one-partition luck.
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
from lgbm_cv import BASE_PARAMS, INTERNAL_VAL_FRAC, fit_full
from step2_familyB import build_features
from step13_temporal import estimate_transition, viterbi, EPS

PARTITION_SEEDS = [42, 7, 123]
NOISE = 0.0024
N_EST_FULL = 463
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

# current best smoother config (LB 0.7867); its 3-partition mean OOF is ~0.7405
CURRENT = dict(s=0.7, e=1.0, beta=0.5, decode="viterbi", trans="emp", prob_mode="log")


def load_train():
    X_raw, meta = D.load_split("train")
    return (build_features(X_raw), meta["label"].to_numpy(),
            meta["file_id"].to_numpy(), meta["user"].to_numpy())


def sticky_transition(T):
    K = len(T); diag = np.diag(T).copy(); Ts = np.empty_like(T)
    for i in range(K):
        Ts[i, :] = (1 - diag[i]) / (K - 1); Ts[i, i] = diag[i]
    return Ts


def forward_backward(log_em, log_T, log_prior):
    """Return normalized log-posteriors (n, K) via sum-product."""
    n, K = log_em.shape
    la = np.empty((n, K)); la[0] = log_prior + log_em[0]
    for t in range(1, n):
        la[t] = log_em[t] + logsumexp(la[t - 1][:, None] + log_T, axis=0)
    lb = np.zeros((n, K))
    for t in range(n - 2, -1, -1):
        lb[t] = logsumexp(log_T + (log_em[t + 1] + lb[t + 1])[None, :], axis=1)
    lp = la + lb
    return lp - logsumexp(lp, axis=1, keepdims=True)


def lgb_oof_probs(X, y, groups, cv, params=None):
    p = TUNED if params is None else params
    oof = np.zeros((len(y), N_CLASSES))
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(X[tr], y[tr], test_size=INTERNAL_VAL_FRAC,
                                          stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X[va])
    return oof


def _emission(probs_idx, prior, e, beta, prob_mode):
    if prob_mode == "log":
        base = np.log(probs_idx + EPS); base_prior = np.log(prior + EPS)
    else:  # raw probabilities (ablation)
        base = probs_idx; base_prior = prior
    return e * (base - beta * base_prior)


def smooth(probs, y, fid, user, groups, cv, s, e, beta, decode, trans, prob_mode,
           return_proba=False):
    pred = np.full(len(y), -1, dtype=int)
    post = np.zeros((len(y), N_CLASSES)) if return_proba else None
    dummy = np.zeros((len(y), 1))
    for tr, va in cv.split(dummy, y, groups):
        T, prior = estimate_transition(y[tr], fid[tr], user[tr])
        if trans == "sticky":
            T = sticky_transition(T)
        log_T = s * np.log(T + EPS); log_prior = s * np.log(prior + EPS)
        for u in np.unique(user[va]):
            idx = va[user[va] == u]; idx = idx[np.argsort(fid[idx])]
            em = _emission(probs[idx], prior, e, beta, prob_mode)
            if decode == "viterbi":
                pred[idx] = viterbi(em, log_T, log_prior)
            else:
                lp = forward_backward(em, log_T, log_prior)
                pred[idx] = lp.argmax(1)
                if return_proba:
                    post[idx] = np.exp(lp)
    return (pred, post) if return_proba else pred


def make_partitions(X, y, groups):
    parts = []
    for ps in PARTITION_SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=ps)
        parts.append((cv, lgb_oof_probs(X, y, groups, cv)))
    return parts


def eval_macro(parts, y, fid, user, groups, **cfg):
    ms = []
    for cv, probs in parts:
        pred = smooth(probs, y, fid, user, groups, cv, **cfg)
        ms.append(f1_score(y, pred, average="macro"))
    return np.array(ms)


def robust_gate(cand_ms, base_ms):
    """Win iff mean gain > noise AND not worse than base on ANY partition."""
    return (cand_ms.mean() - base_ms.mean() > NOISE) and bool(np.all(cand_ms >= base_ms - 1e-9))


# ----- inference -----
def decode_test(probs, fid, user, T, prior, s, e, beta, decode, trans, prob_mode):
    if trans == "sticky":
        T = sticky_transition(T)
    log_T = s * np.log(T + EPS); log_prior = s * np.log(prior + EPS)
    pred = np.full(len(fid), -1, dtype=int)
    for u in np.unique(user):
        idx = np.where(user == u)[0]; idx = idx[np.argsort(fid[idx])]
        em = _emission(probs[idx], prior, e, beta, prob_mode)
        pred[idx] = viterbi(em, log_T, log_prior) if decode == "viterbi" \
            else forward_backward(em, log_T, log_prior).argmax(1)
    return pred


def write_submission(X_tr, y_tr, fid_tr, user_tr, cfg, out_path, X_test=None,
                     class_mult=None):
    import pandas as pd
    from collections import Counter
    full_params = dict(TUNED); full_params["n_estimators"] = N_EST_FULL
    X_test_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    if X_test is None:
        X_test = build_features(X_test_raw)
    model = fit_full(X_tr, y_tr, params=full_params, use_weights=True)
    p_test = model.predict_proba(X_test)
    T, prior = estimate_transition(y_tr, fid_tr, user_tr)
    if class_mult is not None:   # Step B: per-class threshold/bias via FB posteriors
        post = np.zeros((len(fid_te), N_CLASSES))
        if cfg["trans"] == "sticky":
            Tt = sticky_transition(T)
        else:
            Tt = T
        log_T = cfg["s"] * np.log(Tt + EPS); log_prior = cfg["s"] * np.log(prior + EPS)
        for u in np.unique(user_te):
            idx = np.where(user_te == u)[0]; idx = idx[np.argsort(fid_te[idx])]
            em = _emission(p_test[idx], prior, cfg["e"], cfg["beta"], cfg["prob_mode"])
            post[idx] = np.exp(forward_backward(em, log_T, log_prior))
        pred = (post * np.asarray(class_mult)).argmax(1)
    else:
        pred = decode_test(p_test, fid_te, user_te, T, prior, **cfg)

    pred_by_id = dict(zip(fid_te.tolist(), pred.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(out_path, index=False)
    print("Label dist:", {c: Counter(out.tolist()).get(c, 0) for c in range(N_CLASSES)})
    print("Wrote", out_path)
