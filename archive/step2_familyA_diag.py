"""Diagnostic on the Family-A OOF predictions. No new features, no new model.

Recovers the exact same OOF predictions as step2_familyA.py (the pipeline is fully
deterministic), then inspects the confusion matrix with a focus on L2 and L5.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix

import har_data as D
from har_cv import evaluate_oof, N_CLASSES
from step1_harness import lr_factory
from step2_familyA import build_features

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)

# same harness, same seeds -> identical OOF as step2_familyA.py
oof, macro, per_class, _ = evaluate_oof(X, y, groups, lr_factory, stratified=True, verbose=False)
print(f"(sanity) recovered OOF macro-F1 = {macro:.4f}  L2={per_class[2]:.4f}  L5={per_class[5]:.4f}\n")

cm = confusion_matrix(y, oof, labels=list(range(N_CLASSES)))

# ---- 1. full confusion matrix ----
print("1. CONFUSION MATRIX (rows = TRUE, cols = PRED), raw counts")
print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
for t in range(N_CLASSES):
    print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

def breakdown(t):
    row = cm[t]
    n = row.sum()
    print(f"\n{'='*60}\nTRUE-LABEL-{t}: {n} files, predicted as ->")
    print(f"  {'pred':>5} {'count':>7} {'pct':>8}")
    for p in range(N_CLASSES):
        mark = "  <- correct" if p == t else ""
        print(f"  {p:>5} {row[p]:>7} {100*row[p]/n:>7.1f}%{mark}")

# ---- 2 & 3 ----
breakdown(2)
breakdown(5)
