"""STEP B -- multi-model blend: LightGBM + XGBoost + CatBoost (soft vote).

All three trained with equivalent settings: multiclass, inverse-frequency sample
weights, the SAME StratifiedGroupKFold folds, the SAME stratified internal
early-stopping split (random_state=SEED), deterministic (single-threaded, seeded).
We blend OOF class probabilities three ways and compare to the locked 0.7236:
  (a) equal weights (1/3 each)
  (b) LightGBM-heavy (0.5 / 0.25 / 0.25)
  (c) best weights by grid search on OOF macro-F1 (simplex, step 0.05)
Adopt only if a blend beats 0.7236 by > noise floor (0.0024) -> then refit on all
11,020 train files and write submission.

Run: python step11_blend.py
"""
from __future__ import annotations
import json
import itertools
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv
from lgbm_cv import BASE_PARAMS, lgbm_oof_proba, INTERNAL_VAL_FRAC
from step2_familyB import build_features

NOISE = 0.0024
BASE = {"macro": 0.7236, "per_class": [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]}
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission_blend.csv"

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

# Equivalent (sensibly aligned to the tuned LGBM, not separately tuned) settings.
XGB_PARAMS = dict(
    objective="multi:softprob", num_class=N_CLASSES, n_estimators=1000,
    learning_rate=0.0366, max_depth=6, subsample=0.6, colsample_bytree=0.73,
    reg_lambda=0.5, reg_alpha=0.75, min_child_weight=5, gamma=0.1,
    tree_method="hist", random_state=SEED, n_jobs=1, eval_metric="mlogloss",
    early_stopping_rounds=50, verbosity=0,
)
CAT_PARAMS = dict(
    loss_function="MultiClass", iterations=1000, learning_rate=0.0366, depth=6,
    l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8,
    random_seed=SEED, thread_count=1, early_stopping_rounds=50, verbose=0,
)


def xgb_oof_proba(X, y, groups):
    cv = make_cv(stratified=True)
    oof = np.zeros((len(y), N_CLASSES))
    iters = []
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(
            X[tr], y[tr], test_size=INTERNAL_VAL_FRAC, stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xe, ye)], verbose=False)
        bi = m.best_iteration
        oof[va] = m.predict_proba(X[va], iteration_range=(0, bi + 1))
        iters.append(bi)
    return oof, int(np.median(iters))


def cat_oof_proba(X, y, groups):
    cv = make_cv(stratified=True)
    oof = np.zeros((len(y), N_CLASSES))
    iters = []
    for tr, va in cv.split(X, y, groups):
        Xi, Xe, yi, ye = train_test_split(
            X[tr], y[tr], test_size=INTERNAL_VAL_FRAC, stratify=y[tr], random_state=SEED)
        sw = compute_sample_weight("balanced", yi)
        m = CatBoostClassifier(**CAT_PARAMS)
        m.fit(Xi, yi, sample_weight=sw, eval_set=(Xe, ye), use_best_model=True)
        oof[va] = m.predict_proba(X[va])
        iters.append(m.get_best_iteration())
    return oof, int(np.median(iters))


def blend_macro(probs, w):
    P = w[0] * probs[0] + w[1] * probs[1] + w[2] * probs[2]
    return f1_score(y, P.argmax(1), average="macro"), P.argmax(1)


def grid_search_weights(probs):
    best = (-1, None)
    grid = [round(0.05 * i, 2) for i in range(21)]
    for a in grid:
        for b in grid:
            c = round(1 - a - b, 2)
            if c < 0 or c > 1:
                continue
            m, _ = blend_macro(probs, (a, b, c))
            if m > best[0]:
                best = (m, (a, b, c))
    return best


def pc_row(name, m, pred):
    pc = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)))
    print(f"{name:34} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))
    return pc


def detail(tag, pred):
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    p2 = precision_score(y, pred, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, pred, labels=[2], average="micro", zero_division=0)
    p5 = precision_score(y, pred, labels=[5], average="micro", zero_division=0)
    r5 = recall_score(y, pred, labels=[5], average="micro", zero_division=0)
    print(f"  [{tag}] L2 prec={p2:.4f} rec={r2:.4f} | L5 prec={p5:.4f} rec={r5:.4f} | "
          f"true1->pred2={cm[1,2]}")


X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)


def main():
    print(f"STEP B -- LGB + XGB + CAT blend. Baseline 0.7236, noise +/-{NOISE}.\n")
    print("Training OOF probabilities (same folds, weights, internal split)...")
    p_lgb = lgbm_oof_proba(X, y, groups, params=TUNED)
    p_xgb, it_xgb = xgb_oof_proba(X, y, groups)
    p_cat, it_cat = cat_oof_proba(X, y, groups)
    probs = [p_lgb, p_xgb, p_cat]

    m_lgb = f1_score(y, p_lgb.argmax(1), average="macro")
    m_xgb = f1_score(y, p_xgb.argmax(1), average="macro")
    m_cat = f1_score(y, p_cat.argmax(1), average="macro")
    print(f"  single LGB={m_lgb:.4f}  XGB={m_xgb:.4f} (it~{it_xgb})  CAT={m_cat:.4f} (it~{it_cat})")

    m_a, pred_a = blend_macro(probs, (1/3, 1/3, 1/3))
    m_b, pred_b = blend_macro(probs, (0.5, 0.25, 0.25))
    (m_c, w_c) = grid_search_weights(probs)
    _, pred_c = blend_macro(probs, w_c)

    print("\n" + "=" * 90)
    hdr = f"{'model / blend':34} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))
    print(f"{'baseline: LGB single (locked)':34} | {BASE['macro']:7.4f} | "
          + "  | ".join(f"{v:.4f}" for v in BASE["per_class"]))
    pc_row("LGB single (re-run check)", m_lgb, p_lgb.argmax(1))
    pc_row("XGB single", m_xgb, p_xgb.argmax(1))
    pc_row("CAT single", m_cat, p_cat.argmax(1))
    pc_row("(a) equal 1/3 blend", m_a, pred_a)
    pc_row("(b) LGB-heavy .5/.25/.25", m_b, pred_b)
    pc_row(f"(c) grid-best {w_c}", m_c, pred_c)

    print("\nL2/L5 precision+recall:")
    detail("a equal", pred_a); detail("b lgb-heavy", pred_b); detail("c grid", pred_c)

    print(f"\nDeltas vs baseline 0.7236 (noise +/-{NOISE}):")
    best_blend = max([("(a) equal", m_a, pred_a), ("(b) lgb-heavy", m_b, pred_b),
                      ("(c) grid", m_c, pred_c)], key=lambda t: t[1])
    for tag, m, _ in [("(a) equal", m_a, None), ("(b) lgb-heavy", m_b, None), ("(c) grid", m_c, None)]:
        d = m - BASE["macro"]
        v = "WASH" if d <= NOISE else "WIN"
        print(f"  {tag:16}: {m:.4f}  ({d:+.4f})  [{v}]")
    for c in (2, 5):
        bp = best_blend[2]
        pc = f1_score(y, bp, average=None, labels=list(range(N_CLASSES)))
        print(f"  best blend {best_blend[0]} L{c}: {BASE['per_class'][c]:.4f} -> {pc[c]:.4f} "
              f"({pc[c]-BASE['per_class'][c]:+.4f})")

    d_best = best_blend[1] - BASE["macro"]
    if d_best > NOISE:
        print(f"\n>>> {best_blend[0]} beats baseline by >{NOISE} -> writing blend submission")
        write_submission(w_c if best_blend[0].startswith('(c)') else
                         ((1/3,)*3 if best_blend[0].startswith('(a)') else (0.5,0.25,0.25)),
                         it_xgb, it_cat)
    else:
        print(f"\n>>> best blend {best_blend[0]} delta {d_best:+.4f} <= noise -> WASH, "
              f"not adopted. Locked LGB stays.")


def write_submission(w, it_xgb, it_cat):
    X_test_raw, meta_test = D.load_split("test")
    assert "label" not in meta_test.columns
    X_test = build_features(X_test_raw)
    full_lgb = dict(TUNED); full_lgb["n_estimators"] = 463
    xgbp = dict(XGB_PARAMS); xgbp.pop("early_stopping_rounds"); xgbp["n_estimators"] = it_xgb + 1
    catp = dict(CAT_PARAMS); catp.pop("early_stopping_rounds"); catp["iterations"] = it_cat + 1

    def predict_all():
        from lgbm_cv import fit_full
        sw = compute_sample_weight("balanced", y)
        lgbm = fit_full(X, y, params=full_lgb, use_weights=True)
        xm = xgb.XGBClassifier(**xgbp); xm.fit(X, y, sample_weight=sw)
        cm = CatBoostClassifier(**catp); cm.fit(X, y, sample_weight=sw)
        P = w[0]*lgbm.predict_proba(X_test) + w[1]*xm.predict_proba(X_test) + w[2]*cm.predict_proba(X_test)
        return P.argmax(1)

    p1 = predict_all(); p2 = predict_all()
    assert np.array_equal(p1, p2), "blend refit NOT reproducible!"
    print("Reproducibility: two refits -> identical predictions [OK]")
    ids = meta_test["file_id"].to_numpy()
    pred_by_id = dict(zip(ids.tolist(), p1.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out_labels = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out_labels)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out_labels}).to_csv(OUT, index=False)
    print(f"Label dist: {dict(sorted(Counter(out_labels.tolist()).items()))}\nWrote {OUT}")


if __name__ == "__main__":
    main()
