"""Option 1: TabPFN as a DECORRELATED base, blended into the temporal smoother.

XGB/CAT failed to help because they are trees on the same features (correlated errors).
TabPFN is a transformer foundation model for small tabular data -> structurally different
errors. We test whether blending TabPFN into the emission and then applying the validated
robust smoother (Viterbi, empirical transition, beta=0.5, s=0.7) beats the LGB-only
smoother -- validated across 3 fold partitions to guard against overfitting.

Writes submission_tabpfn.csv ONLY if a blend beats the LGB-only smoother by a robust
margin (mean gain > 0.002 across partitions AND not worse on any single partition).

GPU (RTX 3090) used automatically. Run AFTER installing tabpfn (see command below).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, fit_full, INTERNAL_VAL_FRAC
from step2_familyB import build_features
from step13_temporal import estimate_transition, viterbi, EPS

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    raise SystemExit("tabpfn not installed. Run:  pip install tabpfn")

TUNED = dict(BASE_PARAMS); TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
PARTITION_SEEDS = [42, 7, 123]
WEIGHTS = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]  # (LGB, TabPFN)
BETA, S = 0.5, 0.7
DEVICE = "cuda"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission_tabpfn.csv"


def make_tabpfn():
    # balance_probabilities mimics inverse-freq weighting (TabPFN has no sample_weight).
    for kw in (dict(device=DEVICE, random_state=SEED, balance_probabilities=True,
                    ignore_pretraining_limits=True),
               dict(device=DEVICE, random_state=SEED, ignore_pretraining_limits=True),
               dict(device=DEVICE, random_state=SEED)):
        try:
            return TabPFNClassifier(**kw)
        except TypeError:
            continue
    return TabPFNClassifier()


def _align_proba(clf, P):
    out = np.zeros((P.shape[0], N_CLASSES))
    out[:, clf.classes_.astype(int)] = P
    return out


def lgb_oof(X, y, groups, cv):
    oof = np.zeros((len(y), N_CLASSES))
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(X[tr], y[tr], test_size=INTERNAL_VAL_FRAC,
                                          stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = lgb.LGBMClassifier(**TUNED)
        m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X[va])
    return oof


def tabpfn_oof(X, y, groups, cv):
    oof = np.zeros((len(y), N_CLASSES))
    for tr, va in cv.split(X, y, groups):
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        clf = make_tabpfn()
        clf.fit((X[tr] - mu) / sd, y[tr])
        oof[va] = _align_proba(clf, clf.predict_proba((X[va] - mu) / sd))
    return oof


def smooth(proba, y, fid, user, groups, cv):
    out = np.full(len(y), -1, dtype=int)
    dummy = np.zeros((len(y), 1))
    for tr, va in cv.split(dummy, y, groups):
        T, prior = estimate_transition(y[tr], fid[tr], user[tr])
        log_T = S * np.log(T + EPS); log_prior = S * np.log(prior + EPS)
        pc = BETA * np.log(prior + EPS)
        for u in np.unique(user[va]):
            idx = va[user[va] == u]; idx = idx[np.argsort(fid[idx])]
            out[idx] = viterbi(np.log(proba[idx] + EPS) - pc, log_T, log_prior)
    return out


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    groups = user
    X = build_features(X_raw)

    acc = {w: [] for w in WEIGHTS}
    raw_tab, raw_lgb = [], []
    for ps in PARTITION_SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=ps)
        print(f"[partition seed {ps}] computing LGB + TabPFN OOF ...")
        p_lgb = lgb_oof(X, y, groups, cv)
        p_tab = tabpfn_oof(X, y, groups, cv)
        raw_lgb.append(f1_score(y, p_lgb.argmax(1), average="macro"))
        raw_tab.append(f1_score(y, p_tab.argmax(1), average="macro"))
        for (wl, wt) in WEIGHTS:
            pred = smooth(wl * p_lgb + wt * p_tab, y, fid, user, groups, cv)
            acc[(wl, wt)].append(f1_score(y, pred, average="macro"))

    print(f"\nbase argmax (no smooth): LGB mean={np.mean(raw_lgb):.4f}  "
          f"TabPFN mean={np.mean(raw_tab):.4f}")
    print("\nSmoothed macro-F1 by blend weight (across 3 partitions):")
    print(f"  {'LGB':>4} {'Tab':>4} | {'mean':>7} {'std':>6} | per-partition")
    base = None
    rows = []
    for (wl, wt) in WEIGHTS:
        v = np.array(acc[(wl, wt)])
        rows.append(((wl, wt), v.mean(), v.std(), v))
        tag = "  <- LGB-only (current)" if (wl, wt) == (1.0, 0.0) else ""
        print(f"  {wl:>4} {wt:>4} | {v.mean():>7.4f} {v.std():>6.4f} | {[round(x,4) for x in v]}{tag}")
        if (wl, wt) == (1.0, 0.0):
            base = (v.mean(), v)

    best = max(rows, key=lambda r: r[1])
    bmean, bv = best[1], best[3]
    base_mean, base_v = base
    robust_win = (best[0] != (1.0, 0.0) and bmean - base_mean > 0.002
                  and np.all(bv >= base_v - 1e-9))
    print(f"\nbest blend {best[0]}: mean={bmean:.4f}  vs LGB-only {base_mean:.4f}  "
          f"(delta {bmean-base_mean:+.4f})")
    print(f"robust win (mean +>0.002 AND not worse on any partition): {robust_win}")

    if robust_win:
        print(">>> robust improvement -> writing submission_tabpfn.csv")
        write_submission(X, y, fid, user, best[0])
    else:
        print(">>> NOT a robust improvement -> keep current submission_temporal.csv. "
              "TabPFN does not safely beat LGB-only.")


def write_submission(X_train, y_train, fid_tr, user_tr, w):
    wl, wt = w
    full_lgb = dict(TUNED); full_lgb["n_estimators"] = 463
    X_test_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X_test = build_features(X_test_raw)

    lgbm = fit_full(X_train, y_train, params=full_lgb, use_weights=True)
    p_lgb = lgbm.predict_proba(X_test)
    mu = X_train.mean(0); sd = X_train.std(0) + 1e-8
    tab = make_tabpfn(); tab.fit((X_train - mu) / sd, y_train)
    p_tab = _align_proba(tab, tab.predict_proba((X_test - mu) / sd))
    proba = wl * p_lgb + wt * p_tab

    T, prior = estimate_transition(y_train, fid_tr, user_tr)
    log_T = S * np.log(T + EPS); log_prior = S * np.log(prior + EPS); pc = BETA * np.log(prior + EPS)
    pred = np.full(len(fid_te), -1, dtype=int)
    for u in np.unique(user_te):
        idx = np.where(user_te == u)[0]; idx = idx[np.argsort(fid_te[idx])]
        pred[idx] = viterbi(np.log(proba[idx] + EPS) - pc, log_T, log_prior)

    pred_by_id = dict(zip(fid_te.tolist(), pred.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out_labels = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out_labels)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out_labels}).to_csv(OUT, index=False)
    print("Label dist:", {c: Counter(out_labels.tolist()).get(c, 0) for c in range(N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
