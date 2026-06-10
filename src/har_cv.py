"""Cross-validation harness for the HAR competition.

Single source of truth for: the random seed, the CV splitter, and OOF scoring.
Every step imports from here so the validation design never drifts.

CV design (per spec): StratifiedGroupKFold by user, 5 folds. Grouping by user is
mandatory because train/test users are disjoint -- a model must generalize across
people, so validation users must be held out entirely. Stratification keeps rare
classes (e.g. label 4, present in only 19/60 users) balanced across folds so the
per-fold macro-F1 is not dominated by which users happened to land where.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import f1_score

SEED = 42
N_SPLITS = 5
N_CLASSES = 6


def make_cv(stratified: bool = True):
    """Return the CV splitter. StratifiedGroupKFold is the harness we use.

    Both splitters here are deterministic (shuffle disabled / not applicable), so
    no random_state is needed for reproducible folds.
    """
    if stratified:
        return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=False)
    return GroupKFold(n_splits=N_SPLITS)


def evaluate_oof(X, y, groups, model_factory, stratified: bool = True,
                 sample_weight_fn=None, verbose: bool = True):
    """Run grouped CV, return (oof_pred, macro_f1, per_class_f1, fold_f1s).

    model_factory : zero-arg callable returning a fresh, unfitted estimator
                    (so each fold gets an independent model).
    sample_weight_fn : optional callable y_train -> weights, passed to .fit().
    Every file is predicted exactly once, while its user is held out -> leak-free.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    cv = make_cv(stratified=stratified)
    oof = np.full(len(y), -1, dtype=int)
    fold_f1s = []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        model = model_factory()
        if sample_weight_fn is not None:
            model.fit(X[tr], y[tr], sample_weight=sample_weight_fn(y[tr]))
        else:
            model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof[va] = pred
        f = f1_score(y[va], pred, average="macro")
        fold_f1s.append(f)
        if verbose:
            vu = len(set(np.asarray(groups)[va]))
            present = sorted(int(c) for c in set(y[va]))
            print(f"  fold {fold}: macro-F1={f:.4f}  (val users={vu}, labels={present})")

    assert (oof >= 0).all(), "some sample never landed in a validation fold"
    macro = f1_score(y, oof, average="macro")
    per_class = f1_score(y, oof, average=None, labels=list(range(N_CLASSES)))
    if verbose:
        print(f"  mean fold macro-F1: {np.mean(fold_f1s):.4f} "
              f"(+/- {np.std(fold_f1s):.4f})")
        print(f"  OOF macro-F1 (pooled): {macro:.4f}")
        print("  per-class F1: " +
              "  ".join(f"L{c}={per_class[c]:.4f}" for c in range(N_CLASSES)))
    return oof, macro, per_class, fold_f1s


def per_class_table(per_class) -> str:
    return "\n".join(f"    label {c}: F1 = {per_class[c]:.4f}"
                     for c in range(N_CLASSES))
