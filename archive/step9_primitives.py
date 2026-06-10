"""Experimental: bag-of-motion-primitives features (adapted from MoPFormer, NeurIPS 2025).

REPORT EXPERIMENT ONLY. The locked final model (tuned LightGBM, 86 feats, OOF 0.7236)
is NOT touched. This is an additive feature test measured against that exact baseline.

Adaptation: our data is per-second mean/std (not raw waveform), so we cannot use the
raw-signal architecture. We adapt the IDEA -- a codebook of short sub-sequence patterns
("motion primitives") + a per-file bag-of-primitives histogram.

Pipeline (per CV fold, leak-free):
  1. Extract overlapping windows (length w, stride w/2) from each 300x6 file; summarize
     each window by per-channel mean+std (12-dim descriptor).
  2. Fit codebook (MiniBatchKMeans, K primitives) on TRAINING-fold windows ONLY,
     standardized with TRAINING-fold window stats. Assign all windows to primitives.
  3. Per-file normalized histogram over K primitives (+ optional top-T primitive
     transition counts for temporal order).
Three regimes scored with the SAME harness (tuned LightGBM, inverse-freq weights):
  (a) 86 features + histogram
  (b) histogram alone
  (c) 86 + histogram + transitions
Noise floor: column-order/RNG moves OOF ~+/-0.0024; anything inside that is a WASH.

Run: python step9_primitives.py
"""
from __future__ import annotations
import json
import numpy as np
import lightgbm as lgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv
from lgbm_cv import BASE_PARAMS
from step2_familyB import build_features

NOISE = 0.0024
BASE = {"macro": 0.7236, "per_class": [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]}
TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
T_TRANS = 30   # number of most-common primitive transitions kept (regime c)


# ---------- window / codebook helpers ----------
def extract_windows(X_raw, w, stride):
    n, T, C = X_raw.shape
    starts = list(range(0, T - w + 1, stride))
    out = np.empty((n, len(starts), 2 * C), dtype=np.float64)
    for j, s in enumerate(starts):
        seg = X_raw[:, s:s + w, :]
        out[:, j, :C] = seg.mean(axis=1)
        out[:, j, C:] = seg.std(axis=1)
    return out                                   # (n, n_win, 12)


def hist_of(lab, K):
    n, nw = lab.shape
    H = np.zeros((n, K))
    for i in range(n):
        H[i] = np.bincount(lab[i], minlength=K)
    return H / nw


def trans_features(lab, top, K):
    pairs = lab[:, :-1] * K + lab[:, 1:]
    n = lab.shape[0]
    F = np.zeros((n, len(top)))
    for i in range(n):
        F[i] = np.bincount(pairs[i], minlength=K * K)[top]
    return F / (lab.shape[1] - 1)


def train_predict(Xtr, ytr, Xva):
    Xi, Xe, yi, ye = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr,
                                      random_state=SEED)
    sw = compute_sample_weight("balanced", yi)
    m = lgb.LGBMClassifier(**TUNED)
    m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m.predict(Xva)


def oof_all_regimes(X_raw, X86, y, groups, w, K):
    """Return {regime: (oof, macro, per_class, fold_f1s)} for regimes a, b, c."""
    descs = extract_windows(X_raw, w, w // 2)
    dim = descs.shape[2]
    cv = make_cv(stratified=True)
    oof = {r: np.full(len(y), -1, dtype=int) for r in "abc"}
    folds = {r: [] for r in "abc"}

    for tr, va in cv.split(X86, y, groups):
        flat_tr = descs[tr].reshape(-1, dim)
        mu = flat_tr.mean(0); sd = flat_tr.std(0) + 1e-8
        km = MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=3,
                             batch_size=4096, max_iter=100)
        km.fit((flat_tr - mu) / sd)

        def assign(idx):
            d = descs[idx]
            lab = km.predict((d.reshape(-1, dim) - mu) / sd).reshape(d.shape[0], d.shape[1])
            return lab
        lab_tr, lab_va = assign(tr), assign(va)
        Htr, Hva = hist_of(lab_tr, K), hist_of(lab_va, K)
        top = np.argsort(np.bincount((lab_tr[:, :-1] * K + lab_tr[:, 1:]).ravel(),
                                     minlength=K * K))[::-1][:T_TRANS]
        Ttr, Tva = trans_features(lab_tr, top, K), trans_features(lab_va, top, K)

        feats = {
            "a": (np.hstack([X86[tr], Htr]), np.hstack([X86[va], Hva])),
            "b": (Htr, Hva),
            "c": (np.hstack([X86[tr], Htr, Ttr]), np.hstack([X86[va], Hva, Tva])),
        }
        for r, (Xtr_, Xva_) in feats.items():
            pred = train_predict(Xtr_, y[tr], Xva_)
            oof[r][va] = pred
            folds[r].append(f1_score(y[va], pred, average="macro"))

    res = {}
    for r in "abc":
        macro = f1_score(y, oof[r], average="macro")
        pc = f1_score(y, oof[r], average=None, labels=list(range(N_CLASSES)))
        res[r] = (oof[r], macro, pc, folds[r])
    return res


def detail(tag, y, oof):
    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    p2 = precision_score(y, oof, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, oof, labels=[2], average="micro", zero_division=0)
    p5 = precision_score(y, oof, labels=[5], average="micro", zero_division=0)
    r5 = recall_score(y, oof, labels=[5], average="micro", zero_division=0)
    print(f"  [{tag}] L2 prec={p2:.4f} rec={r2:.4f} | L5 prec={p5:.4f} rec={r5:.4f} | "
          f"true1->pred2={cm[1,2]}")


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    print(f"Motion-primitives experiment. Baseline (86 feats, tuned LGBM) = {BASE['macro']:.4f}")
    print(f"Noise floor +/-{NOISE}. Per-fold codebook refit (leak-free).\n")

    configs = [(10, 32), (10, 64), (10, 128), (20, 32), (20, 64), (20, 128)]
    print("SWEEP -- regime (a) 86+histogram, by (w, K):")
    print(f"  {'w':>3} {'K':>4} | {'macro':>7} {'dmac':>8} | {'L2':>6} {'L5':>6} | fold_std")
    sweep = {}
    for w, K in configs:
        res = oof_all_regimes(X_raw, X86, y, groups, w, K)
        sweep[(w, K)] = res
        _, m, pc, fs = res["a"]
        print(f"  {w:>3} {K:>4} | {m:>7.4f} {m-BASE['macro']:>+8.4f} | "
              f"{pc[2]:>6.4f} {pc[5]:>6.4f} | {np.std(fs):.4f}")

    best_wK = max(configs, key=lambda c: sweep[c]["a"][1])
    print(f"\nBest codebook for regime (a): w={best_wK[0]}, K={best_wK[1]} "
          f"(macro {sweep[best_wK]['a'][1]:.4f})")
    res = sweep[best_wK]

    print("\n" + "=" * 90)
    hdr = f"{'regime':40} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        print(f"{name:40} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))

    row("baseline: 86 feats (locked)", BASE["macro"], BASE["per_class"])
    row(f"(a) 86 + histogram  [w{best_wK[0]} K{best_wK[1]}]", res["a"][1], res["a"][2])
    row(f"(b) histogram alone [w{best_wK[0]} K{best_wK[1]}]", res["b"][1], res["b"][2])
    row(f"(c) 86 + hist + transitions", res["c"][1], res["c"][2])

    print("\nL2/L5 precision+recall:")
    for r, tag in [("a", "a 86+hist"), ("b", "b hist-only"), ("c", "c 86+hist+trans")]:
        detail(tag, y, res[r][0])

    print("\nPer-fold spread + delta vs baseline (noise floor +/-{:.4f}):".format(NOISE))
    for r, tag in [("a", "(a) 86+hist"), ("b", "(b) hist-only"), ("c", "(c) 86+hist+trans")]:
        _, m, _, fs = res[r]
        d = m - BASE["macro"]
        verdict = "WASH (within noise)" if abs(d) <= NOISE else ("WIN" if d > 0 else "LOSS")
        print(f"  {tag:20}: macro {m:.4f}  delta {d:+.4f}  [{verdict}]  "
              f"folds={[round(f,3) for f in fs]} std={np.std(fs):.4f}")


if __name__ == "__main__":
    main()
