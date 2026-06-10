"""Step 2 -- Family B (temporal / rhythm) ONLY.

Adds temporal-arrangement features ON TOP OF the 61 Family-A features (12 baseline +
49 distribution-shape). Same StratifiedGroupKFold harness, same seeds, and the
balanced LogisticRegression (class_weight='balanced') which is now the default model.
We stay on LR so this measures FEATURES, not a model swap.

Baseline row to beat (balanced model, Family A only):
  macro 0.6308 | L0 0.9100 L1 0.7557 L2 0.2287 L3 0.6456 L4 0.7492 L5 0.4957

Family B is computed on the per-file motion-magnitude sequence
  mag[t] = sqrt(std_x[t]^2 + std_y[t]^2 + std_z[t]^2),  t = 0..299,
plus a sensible rhythm subset on each individual std_* axis. All thresholds are the
file's OWN median/mean (self-normalizing -- no global constant).

Run: python step2_familyB.py
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, evaluate_oof, N_CLASSES
from step2_familyA import build_features as build_A          # 61 feats
from step2_familyA_weighted import lr_balanced_factory       # the default model

np.random.seed(SEED)
EPS = 1e-9
STD_IDX = [3, 4, 5]   # std_x, std_y, std_z in D.SENSOR_COLS order

BALANCED_A = {
    "macro": 0.6308,
    "per_class": [0.9100, 0.7557, 0.2287, 0.6456, 0.7492, 0.4957],
}


# ---------- vectorized helpers (operate on (n, T) arrays) ----------
def _crossing_rate(x, thr):
    """Fraction of adjacent steps where x crosses the per-file threshold thr."""
    b = x > thr                                   # (n, T) bool
    return (b[:, 1:] != b[:, :-1]).mean(axis=1)   # (n,)


def _active_fraction(x, thr):
    return (x > thr).mean(axis=1)


def _autocorr(x, lag):
    """Per-file autocorrelation at `lag` (overall-mean / overall-variance form)."""
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    num = (xc[:, lag:] * xc[:, :-lag]).sum(axis=1)
    den = (xc * xc).sum(axis=1) + EPS
    return num / den


def _longest_true_run_1d(mask):
    if not mask.any():
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


# ---------- Family B feature builder ----------
def family_B_features(X_raw: np.ndarray) -> np.ndarray:
    n, T, _ = X_raw.shape
    mag = np.sqrt((X_raw[:, :, STD_IDX] ** 2).sum(axis=2))     # (n, T)

    med = np.median(mag, axis=1, keepdims=True)                # (n,1)
    mean = mag.mean(axis=1, keepdims=True)                     # (n,1)

    # --- vectorized rhythm stats on mag ---
    cross_rate = _crossing_rate(mag, med)                      # (n,)
    active_med = _active_fraction(mag, med)
    active_mean = _active_fraction(mag, mean)
    ac = [_autocorr(mag, k) for k in range(1, 6)]              # 5 x (n,)
    argmax_norm = mag.argmax(axis=1) / (T - 1)                 # temporal location of peak
    energy = mag ** 2
    first_half_frac = energy[:, : T // 2].sum(axis=1) / (energy.sum(axis=1) + EPS)

    # --- per-file loop for run lengths + peak count ---
    longest_active = np.empty(n)
    longest_inactive = np.empty(n)
    peak_count = np.empty(n)
    mag_std = mag.std(axis=1)
    for i in range(n):
        m = mag[i]
        thr = med[i, 0]
        longest_active[i] = _longest_true_run_1d(m > thr)
        longest_inactive[i] = _longest_true_run_1d(m <= thr)
        prom = max(EPS, 0.5 * mag_std[i])     # simple self-normalizing prominence
        peaks, _ = find_peaks(m, prominence=prom)
        peak_count[i] = len(peaks)

    mag_block = np.column_stack([
        cross_rate, active_med, active_mean,
        longest_active, longest_inactive,
        ac[0], ac[1], ac[2], ac[3], ac[4],
        peak_count, argmax_norm, first_half_frac,
    ])  # 13 features

    # --- rhythm subset on each std_* axis (4 each = 12) ---
    axis_blocks = []
    for k in STD_IDX:
        ax = X_raw[:, :, k]
        amed = np.median(ax, axis=1, keepdims=True)
        cr = _crossing_rate(ax, amed)
        af = _active_fraction(ax, amed)
        a1 = _autocorr(ax, 1)
        lar = np.array([_longest_true_run_1d(ax[i] > amed[i, 0]) for i in range(n)])
        axis_blocks.append(np.column_stack([cr, af, a1, lar]))

    return np.concatenate([mag_block] + axis_blocks, axis=1)   # 13 + 12 = 25


def build_features(X_raw: np.ndarray) -> np.ndarray:
    """Family A (61) + Family B (25) = 86 features."""
    return np.concatenate([build_A(X_raw), family_B_features(X_raw)], axis=1)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X = build_features(X_raw)
    print(f"feature matrix: Family A 61 + Family B 25 = {X.shape[1]} features\n")

    print("[harness] StratifiedGroupKFold, LogisticRegression(class_weight='balanced'), seed=42:")
    oof, macro, per_class, _ = evaluate_oof(X, y, groups, lr_balanced_factory, stratified=True)

    # ---- comparison table ----
    print("\n" + "=" * 78)
    hdr = f"{'feature_set':22} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        cells = "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES))
        print(f"{name:22} | {m:7.4f} | {cells}")

    row("Family A (balanced)", BALANCED_A["macro"], BALANCED_A["per_class"])
    row("+ Family B", macro, per_class)

    # ---- explicit deltas ----
    print("\nDeltas vs balanced Family-A baseline:")
    for c in (2, 5, 1):
        b, nw = BALANCED_A["per_class"][c], per_class[c]
        print(f"  L{c} F1: {b:.4f} -> {nw:.4f} ({nw - b:+.4f})")
    print(f"  macro-F1: {BALANCED_A['macro']:.4f} -> {macro:.4f} ({macro - BALANCED_A['macro']:+.4f})")

    # ---- confusion matrix ----
    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    print("\nCONFUSION MATRIX, +Family B (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
    print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

    # ---- the key cell + L2 precision/recall ----
    t1_p2 = cm[1, 2]
    l2_prec = precision_score(y, oof, labels=[2], average="micro", zero_division=0)
    l2_rec = recall_score(y, oof, labels=[2], average="micro", zero_division=0)
    print("\nL2-vs-L1 separation check:")
    print(f"  true1 -> pred2 false positives: {t1_p2}   (was 678 under Family A balanced)")
    print(f"  L2 precision: {l2_prec:.4f}   (TP={cm[2,2]}, predicted-2 total={cm[:,2].sum()})")
    print(f"  L2 recall   : {l2_rec:.4f}   (TP={cm[2,2]}, true-2 total={cm[2].sum()})")
    print(f"  L1 recall   : {cm[1,1] / cm[1].sum():.4f}   (true1->pred1 = {cm[1,1]} of {cm[1].sum()})")


if __name__ == "__main__":
    main()
