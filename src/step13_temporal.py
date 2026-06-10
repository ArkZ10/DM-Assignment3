"""Temporal post-processing: HMM/Viterbi smoothing over each user's file sequence.

KEY DATA FACT (verified): files are consecutive 5-min windows -- within a user, file_ids
are contiguous (0 gaps) and adjacent files share a label 89.1% of the time (chance 36.6%).
Every prior model predicted files independently, ignoring this. Here we decode each
user's label sequence with Viterbi: emissions = the locked LightGBM's leak-free OOF class
probabilities; transitions = the label-transition matrix estimated from TRAINING-fold
sequences only. Leak-free: OOF probs already hold the user out, and T is fit per fold on
train users only. Test file ordering is given, so this is legitimate (no label leakage).

Smoothing strength `s` scales log-transition + log-prior. s=0 -> pure emission argmax
(reproduces the 0.7236 base model); larger s -> stickier sequences. We sweep s and keep
the best, adopting only if it beats 0.7236 by > noise floor (0.0024).

Run: python step13_temporal.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv
from lgbm_cv import BASE_PARAMS, lgbm_oof_proba, fit_full
from step2_familyB import build_features

NOISE = 0.0024
BASE_MACRO = 0.7236
BASE_PC = [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]
S_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]
BETA_GRID = [0.0, 0.5, 1.0]
EPS = 1e-12
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission_temporal.csv"

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))


def estimate_transition(y_seq, fid_seq, user_seq, lap=1.0):
    """Row-normalized transition matrix + prior from per-user file_id-ordered label runs."""
    T = np.full((N_CLASSES, N_CLASSES), lap)
    prior = np.full(N_CLASSES, lap)
    for u in np.unique(user_seq):
        m = user_seq == u
        order = np.argsort(fid_seq[m])
        yy = y_seq[m][order]
        for c in yy:
            prior[c] += 1
        for a, b in zip(yy[:-1], yy[1:]):
            T[a, b] += 1
    T = T / T.sum(axis=1, keepdims=True)
    prior = prior / prior.sum()
    return T, prior


def viterbi(log_em, log_T, log_prior):
    n, K = log_em.shape
    dp = np.empty((n, K)); back = np.empty((n, K), dtype=int)
    dp[0] = log_prior + log_em[0]
    for t in range(1, n):
        scores = dp[t - 1][:, None] + log_T          # (K_prev, K_next)
        back[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_em[t]
    path = np.empty(n, dtype=int)
    path[-1] = dp[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_oof(oof_proba, y, fid, user, groups, s, beta=0.0):
    """Apply per-fold-trained Viterbi smoothing to the OOF probabilities (leak-free).

    s    : smoothing strength (scales log-transition + log-prior). s=0 -> base argmax.
    beta : prior correction. emission = log P(c|x) - beta*log P(c); beta=0 uses the raw
           posterior, beta=1 the full scaled likelihood P(x|c). beta protects rare,
           non-sticky classes (L2) from being absorbed by sticky neighbors (L1).
    """
    cv = make_cv(stratified=True)
    dummyX = np.zeros((len(y), 1))
    out = np.full(len(y), -1, dtype=int)
    for tr, va in cv.split(dummyX, y, groups):
        T, prior = estimate_transition(y[tr], fid[tr], user[tr])
        log_T = s * np.log(T + EPS)
        log_prior = s * np.log(prior + EPS)
        prior_corr = beta * np.log(prior + EPS)
        for u in np.unique(user[va]):
            idx = va[user[va] == u]
            idx = idx[np.argsort(fid[idx])]          # time order
            em = np.log(oof_proba[idx] + EPS) - prior_corr
            out[idx] = viterbi(em, log_T, log_prior)
    return out


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    fid = meta["file_id"].to_numpy()
    user = meta["user"].to_numpy()
    groups = user
    X = build_features(X_raw)

    print(f"Temporal Viterbi smoothing. Base 0.7236, noise +/-{NOISE}.\n")
    print("Computing base OOF probabilities (locked tuned LightGBM)...")
    oof_p = lgbm_oof_proba(X, y, groups, params=TUNED)
    base_macro = f1_score(y, oof_p.argmax(1), average="macro")
    print(f"  base OOF macro-F1 (argmax): {base_macro:.4f}\n")

    # report transition diagonal for interpretability (full-data estimate)
    Tfull, _ = estimate_transition(y, fid, user)
    print(f"transition diagonal (P stay): {[round(Tfull[c,c],3) for c in range(N_CLASSES)]}\n")

    print("Smoothing sweep over prior-correction beta x strength s:")
    print(f"  {'beta':>4} {'s':>4} | {'macroF1':>7} | L2     L5")
    results = {}
    for beta in BETA_GRID:
        for s in S_GRID:
            pred = smooth_oof(oof_p, y, fid, user, groups, s, beta)
            macro = f1_score(y, pred, average="macro")
            pc = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)))
            results[(beta, s)] = (macro, pc, pred)
            tag = "  <- (0,0) reproduces base" if (beta == 0 and s == 0) else ""
            print(f"  {beta:>4} {s:>4} | {macro:>7.4f} | {pc[2]:.4f} {pc[5]:.4f}{tag}")

    (best_beta, best_s) = max(results, key=lambda k: results[k][0])
    bmacro, bpc, bpred = results[(best_beta, best_s)]

    print("\n" + "=" * 86)
    hdr = f"{'model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))

    row("base LightGBM (locked)", BASE_MACRO, BASE_PC)
    row(f"+ Viterbi (beta={best_beta}, s={best_s})", bmacro, bpc)

    d = bmacro - BASE_MACRO
    verdict = "WASH (<= noise)" if d <= NOISE else "WIN (> noise)"
    print(f"\nL2: {BASE_PC[2]:.4f} -> {bpc[2]:.4f} ({bpc[2]-BASE_PC[2]:+.4f})")
    print(f"L5: {BASE_PC[5]:.4f} -> {bpc[5]:.4f} ({bpc[5]-BASE_PC[5]:+.4f})")
    print(f"macro: {BASE_MACRO:.4f} -> {bmacro:.4f} ({d:+.4f})  [{verdict}]")

    for c in (2, 5):
        p = precision_score(y, bpred, labels=[c], average="micro", zero_division=0)
        r = recall_score(y, bpred, labels=[c], average="micro", zero_division=0)
        print(f"  L{c}: precision={p:.4f} recall={r:.4f}")

    cm = confusion_matrix(y, bpred, labels=list(range(N_CLASSES)))
    print(f"\nCONFUSION MATRIX (beta={best_beta}, s={best_s})  (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")

    if d > NOISE:
        print(f"\n>>> beats base by > noise floor -> writing temporally-smoothed submission")
        write_submission(X, y, fid, user, best_s, best_beta)
    else:
        print(f"\n>>> WASH -> not adopted.")


def write_submission(X_train, y_train, fid_train, user_train, s, beta):
    full_params = dict(TUNED); full_params["n_estimators"] = 463
    X_test_raw, meta_test = D.load_split("test")
    assert "label" not in meta_test.columns
    fid_te = meta_test["file_id"].to_numpy()
    user_te = meta_test["user"].to_numpy()
    X_test = build_features(X_test_raw)

    # test contiguity check (informational)
    gaps = 0
    for u in np.unique(user_te):
        f = np.sort(fid_te[user_te == u])
        gaps += int((np.diff(f) != 1).sum())
    print(f"test within-user file_id gaps: {gaps} (0 = perfectly contiguous like train)")

    # transition from ALL train labels
    T, prior = estimate_transition(y_train, fid_train, user_train)
    log_T = s * np.log(T + EPS); log_prior = s * np.log(prior + EPS)
    prior_corr = beta * np.log(prior + EPS)

    def predict_all():
        model = fit_full(X_train, y_train, params=full_params, use_weights=True)
        proba = model.predict_proba(X_test)
        pred = np.full(len(fid_te), -1, dtype=int)
        for u in np.unique(user_te):
            idx = np.where(user_te == u)[0]
            idx = idx[np.argsort(fid_te[idx])]
            pred[idx] = viterbi(np.log(proba[idx] + EPS) - prior_corr, log_T, log_prior)
        return pred

    p1 = predict_all(); p2 = predict_all()
    assert np.array_equal(p1, p2), "temporal refit NOT reproducible!"
    print("Reproducibility: two refits -> identical predictions [OK]")

    pred_by_id = dict(zip(fid_te.tolist(), p1.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out_labels = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out_labels)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out_labels}).to_csv(OUT, index=False)
    dist = Counter(out_labels.tolist())
    print("Label distribution:", {c: dist.get(c, 0) for c in range(N_CLASSES)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
