"""Class-weighting check on the EXACT Family-A feature set (61 feats). No new features.

Same StratifiedGroupKFold harness, same seeds. Only change: LogisticRegression with
class_weight='balanced' vs the unweighted Family-A run. This is an ablation row for
report Q4 -- judged on rare-class (L2/L5) recovery vs the L1 cost, NOT the macro-F1
headline (macro can fall even when balancing is the right move).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import har_data as D
from har_cv import SEED, evaluate_oof, N_CLASSES
from step2_familyA import build_features

np.random.seed(SEED)

# Family-A unweighted reference (from step2_familyA.py)
UNWEIGHTED = {
    "macro": 0.6113,
    "per_class": [0.9136, 0.8313, 0.0254, 0.6590, 0.7897, 0.4488],
}


def lr_balanced_factory():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"),
    )


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X = build_features(X_raw)
    print(f"feature set: Family A, {X.shape[1]} features (unchanged)\n")

    print("[harness] StratifiedGroupKFold, LogisticRegression(class_weight='balanced'), seed=42:")
    oof, macro, per_class, _ = evaluate_oof(X, y, groups, lr_balanced_factory, stratified=True)

    # ---- 2. comparison table ----
    print("\n" + "=" * 78)
    hdr = f"{'feature_set':28} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr)
    print("-" * len(hdr))

    def row(name, m, pc):
        cells = "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES))
        print(f"{name:28} | {m:7.4f} | {cells}")

    row("Family A (unweighted)", UNWEIGHTED["macro"], UNWEIGHTED["per_class"])
    row("Family A (balanced weights)", macro, per_class)

    # ---- 3. confusion matrix for balanced run ----
    cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))
    print("\n3. CONFUSION MATRIX, balanced run (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
    print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

    # ---- 4. key deltas ----
    print("\n4. Key per-class deltas (balanced - unweighted):")
    for c in (2, 5, 1):
        b, n = UNWEIGHTED["per_class"][c], per_class[c]
        print(f"  L{c}: {b:.4f} -> {n:.4f} ({n - b:+.4f})")
    print(f"  macro-F1: {UNWEIGHTED['macro']:.4f} -> {macro:.4f} ({macro - UNWEIGHTED['macro']:+.4f})")


if __name__ == "__main__":
    main()
