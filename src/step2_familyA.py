"""Step 2 -- Family A (distribution shape) ONLY.

Adds distribution-shape features ON TOP OF the 12 baseline mean+std features and
re-runs the SAME StratifiedGroupKFold harness with the SAME LogisticRegression and
SAME seeds, so the comparison isolates the FEATURES (model is held fixed; we switch
to LightGBM only after all feature families are measured).

Baseline for the additive table is the StratifiedGroupKFold score 0.5698.
NOTE: the 0.548 -> 0.570 change vs the audit is a CV-DESIGN CORRECTION (GroupKFold ->
StratifiedGroupKFold), i.e. a more honest estimate, NOT a model/feature improvement.
All technique deltas below are measured against 0.5698.

Family A: for each of the 6 sensor columns AND a derived per-timestep motion
magnitude mag = sqrt(std_x^2 + std_y^2 + std_z^2), compute over the file's 300 rows:
  p10, p25, p50, p75, p90, IQR (=p75-p25), max.
7 stats x 7 series = 49 features, added to the 12 baseline -> 61 total.

Run: python step2_familyA.py
"""
from __future__ import annotations
import numpy as np

import har_data as D
from har_cv import SEED, evaluate_oof, N_CLASSES
from step1_harness import baseline_12_features, lr_factory

np.random.seed(SEED)

# Locked-in StratifiedGroupKFold baseline (from Step 1), for the additive table.
BASELINE = {
    "macro": 0.5698,
    "per_class": [0.8948, 0.7983, 0.0319, 0.6606, 0.8015, 0.2318],
}

# std_* column indices in D.SENSOR_COLS order (mean_x,mean_y,mean_z,std_x,std_y,std_z)
STD_IDX = [3, 4, 5]


def _shape_stats(series: np.ndarray) -> np.ndarray:
    """series: (n, T). Return (n, 7): p10,p25,p50,p75,p90, IQR, max."""
    p10, p25, p50, p75, p90 = np.percentile(series, [10, 25, 50, 75, 90], axis=1)
    iqr = p75 - p25
    mx = series.max(axis=1)
    return np.stack([p10, p25, p50, p75, p90, iqr, mx], axis=1)


def family_A_features(X_raw: np.ndarray) -> np.ndarray:
    """49 distribution-shape features: 7 stats over each of 6 columns + mag."""
    blocks = []
    for k in range(X_raw.shape[2]):            # each of the 6 sensor columns
        blocks.append(_shape_stats(X_raw[:, :, k]))
    mag = np.sqrt((X_raw[:, :, STD_IDX] ** 2).sum(axis=2))  # (n, 300) motion magnitude
    blocks.append(_shape_stats(mag))
    return np.concatenate(blocks, axis=1)      # (n, 49)


def build_features(X_raw: np.ndarray) -> np.ndarray:
    """Baseline 12 + Family A 49 = 61 features."""
    return np.concatenate([baseline_12_features(X_raw), family_A_features(X_raw)], axis=1)


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()

    X = build_features(X_raw)
    print(f"feature matrix: baseline 12 + Family A 49 = {X.shape[1]} features\n")

    print("[harness] StratifiedGroupKFold, LogisticRegression, seed=42:")
    _, macro, per_class, _ = evaluate_oof(X, y, groups, lr_factory, stratified=True)

    # ---------- comparison table ----------
    print("\n" + "=" * 78)
    hdr = f"{'feature_set':22} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr)
    print("-" * len(hdr))

    def row(name, macro_v, pc):
        cells = "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES))
        print(f"{name:22} | {macro_v:7.4f} | {cells}")

    row("baseline (12 feats)", BASELINE["macro"], BASELINE["per_class"])
    row("+ Family A (61 feats)", macro, per_class)

    # ---------- the deltas that matter most: L2 and L5 ----------
    print("\nBottleneck-class deltas (the numbers that matter most):")
    for c in (2, 5):
        base, new = BASELINE["per_class"][c], per_class[c]
        print(f"  L{c}: {base:.4f} -> {new:.4f} ({new - base:+.4f})")
    print(f"  macro-F1: {BASELINE['macro']:.4f} -> {macro:.4f} ({macro - BASELINE['macro']:+.4f})")

    # ---------- automated read of the decision rule ----------
    d2 = per_class[2] - BASELINE["per_class"][2]
    d5 = per_class[5] - BASELINE["per_class"][5]
    dmac = macro - BASELINE["macro"]
    print("\nRead (per the decision rule):")
    if d2 > 0.05 and d5 > 0.05:
        verdict = ("(a) Both L2 and L5 moved >+0.05 -- distribution features ARE working; "
                   "continue to Family B to push further.")
    elif d2 < 0.02 and d5 < 0.02 and dmac > 0:
        verdict = ("(b) L2/L5 barely moved (<+0.02) but macro-F1 rose -- gains are in "
                   "already-easy classes; distribution shape is NOT cracking the bottleneck. "
                   "Pivot to temporal features (Family B) as the real fix; don't over-invest here.")
    elif abs(dmac) < 1e-3 and abs(d2) < 1e-3 and abs(d5) < 1e-3:
        verdict = "(c) Nothing moved -- FLAG: something's off."
    else:
        verdict = (f"Mixed/in-between: L2 {d2:+.4f}, L5 {d5:+.4f}, macro {dmac:+.4f} -- "
                   "see explicit reasoning below (does not cleanly fit a/b/c).")
    print("  " + verdict)


if __name__ == "__main__":
    main()
