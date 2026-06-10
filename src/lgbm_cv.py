"""Reusable LightGBM CV core -- single source of truth for the OOF harness.

Replicates step3_lightgbm.py EXACTLY (same outer StratifiedGroupKFold folds, same
stratified internal early-stopping split, same inverse-frequency sample weights), but
parameterized by a params dict so tuning / feature-selection / SMOTE steps all share
one honest, leak-free harness. Default params here reproduce the 0.7095 baseline.
"""
from __future__ import annotations
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

from har_cv import SEED, N_CLASSES, make_cv

warnings.filterwarnings("ignore", message="X does not have valid feature names")

EARLY_STOP = 50
INTERNAL_VAL_FRAC = 0.15

# Step-3 baseline params (reproduce OOF macro-F1 0.7095 on the 86-feature A+B set).
BASE_PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    n_jobs=1,
    verbose=-1,
)


def lgbm_oof(X, y, groups, params=None, use_weights=True, resampler=None,
             weight_fn=None, return_models=False):
    """Run the StratifiedGroupKFold OOF harness.

    use_weights : inverse-frequency sample weights on the (resampled) train set.
    resampler   : optional callable (X_tr, y_tr) -> (X_res, y_res), fit on TRAIN
                  rows only, inside the fold (for SMOTE -- never touches val).
    weight_fn   : optional callable y_train -> sample_weights, overriding the default
                  'balanced' scheme (used by the L2/L5 weight-strength sweep).
    Returns (oof, macro, per_class, fold_f1s, best_iters).
    """
    p = dict(BASE_PARAMS if params is None else params)
    X = np.asarray(X); y = np.asarray(y)
    cv = make_cv(stratified=True)
    oof = np.full(len(y), -1, dtype=int)
    fold_f1s, best_iters = [], []

    for tr, va in cv.split(X, y, groups):
        X_tr, y_tr = X[tr], y[tr]
        # internal early-stopping split (stratified, seeded) -- on real train rows,
        # BEFORE any resampling, so the early-stop signal is on genuine data.
        Xi, Xe, yi, ye = train_test_split(
            X_tr, y_tr, test_size=INTERNAL_VAL_FRAC, stratify=y_tr, random_state=SEED)
        if resampler is not None:
            Xi, yi = resampler(Xi, yi)
        if weight_fn is not None:
            sw = weight_fn(yi)
        elif use_weights:
            sw = compute_sample_weight("balanced", yi)
        else:
            sw = None

        model = lgb.LGBMClassifier(**p)
        model.fit(Xi, yi, sample_weight=sw,
                  eval_set=[(Xe, ye)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(0)])
        oof[va] = model.predict(X[va])
        fold_f1s.append(f1_score(y[va], oof[va], average="macro"))
        best_iters.append(model.best_iteration_)

    macro = f1_score(y, oof, average="macro")
    per_class = f1_score(y, oof, average=None, labels=list(range(N_CLASSES)))
    return oof, macro, per_class, fold_f1s, best_iters


def lgbm_oof_proba(X, y, groups, params=None, use_weights=True):
    """Same harness as lgbm_oof but returns OOF class PROBABILITIES (n, N_CLASSES).

    Used by the seed ensemble and the multi-model blend (soft voting). Each row is the
    held-out prediction for that file, while its user was out of training -> leak-free.
    """
    p = dict(BASE_PARAMS if params is None else params)
    X = np.asarray(X); y = np.asarray(y)
    cv = make_cv(stratified=True)
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(
            X[tr], y[tr], test_size=INTERNAL_VAL_FRAC, stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi) if use_weights else None
        model = lgb.LGBMClassifier(**p)
        model.fit(Xi, yi, sample_weight=sw,
                  eval_set=[(Xe, ye)], eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(0)])
        oof[va] = model.predict_proba(X[va])
    return oof


def fit_full(X, y, params=None, use_weights=True, resampler=None, weight_fn=None):
    """Refit on ALL training rows (no holdout) for inference. Returns fitted model.

    n_estimators is fixed (median best_iteration learned in CV is passed via params),
    so no early-stopping/holdout is needed here.
    """
    p = dict(BASE_PARAMS if params is None else params)
    X = np.asarray(X); y = np.asarray(y)
    if resampler is not None:
        X, y = resampler(X, y)
    if weight_fn is not None:
        sw = weight_fn(y)
    elif use_weights:
        sw = compute_sample_weight("balanced", y)
    else:
        sw = None
    model = lgb.LGBMClassifier(**p)
    model.fit(X, y, sample_weight=sw)
    return model
