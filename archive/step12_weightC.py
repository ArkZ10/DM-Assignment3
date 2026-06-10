"""STEP C -- L2/L5 weight-strength sweep (precision/recall slide).

A knob, not a model change. On the locked tuned LightGBM (Steps A and B were washes, so
the best config IS the locked single LGB), we scale the sample weights of classes 2 and
5 by multipliers m2, m5 ON TOP OF the 'balanced' weights. m=1.0 reproduces the 0.7236
baseline; higher m pushes recall up / precision down for that class. Sweep the requested
multipliers {1.0, 1.5, 2, 3, 5} on classes 2 and 5; report which maximizes OOF macro-F1.
Adopt only if it beats 0.7236 by > noise floor (0.0024).

Run: python step12_weightC.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof, fit_full
from step2_familyB import build_features

NOISE = 0.0024
BASE_MACRO = 0.7236
BASE_PC = [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]
MULTS = [1.0, 1.5, 2.0, 3.0, 5.0]
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission_weighted.csv"

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)


def make_weight_fn(m2, m5):
    def wf(y_tr):
        w = compute_sample_weight("balanced", y_tr)
        w = w.copy()
        w[y_tr == 2] *= m2
        w[y_tr == 5] *= m5
        return w
    return wf


def main():
    print(f"STEP C -- L2/L5 weight-strength sweep. Baseline {BASE_MACRO}, noise +/-{NOISE}.")
    print("Multipliers on classes {2,5} on top of 'balanced'. m=1.0 -> baseline.\n")

    grid = {}
    print("macro-F1 grid (rows=m2, cols=m5):")
    print("       " + "".join(f"m5={m:<6}" for m in MULTS))
    for m2 in MULTS:
        cells = []
        for m5 in MULTS:
            _, macro, pc, fs, _ = lgbm_oof(X, y, groups, params=TUNED,
                                           weight_fn=make_weight_fn(m2, m5))
            grid[(m2, m5)] = (macro, pc, fs)
            cells.append(f"{macro:.4f}")
        print(f"m2={m2:<4} " + " ".join(cells))

    # sanity: (1.0,1.0) must reproduce baseline
    base_check = grid[(1.0, 1.0)][0]
    print(f"\n(1.0,1.0) reproduces baseline: {base_check:.4f} (expect 0.7236)")

    best = max(grid.items(), key=lambda kv: kv[1][0])
    (bm2, bm5), (bmacro, bpc, bfs) = best
    print(f"\nBest combo: m2={bm2}, m5={bm5}  ->  macro {bmacro:.4f}")

    print("\n" + "=" * 86)
    hdr = f"{'config':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))

    row("baseline (m2=1, m5=1)", BASE_MACRO, BASE_PC)
    row(f"best (m2={bm2}, m5={bm5})", bmacro, bpc)

    d = bmacro - BASE_MACRO
    verdict = "WASH (<= noise)" if d <= NOISE else "WIN (> noise)"
    print(f"\nL2: {BASE_PC[2]:.4f} -> {bpc[2]:.4f} ({bpc[2]-BASE_PC[2]:+.4f})")
    print(f"L5: {BASE_PC[5]:.4f} -> {bpc[5]:.4f} ({bpc[5]-BASE_PC[5]:+.4f})")
    print(f"macro: {BASE_MACRO:.4f} -> {bmacro:.4f} ({d:+.4f})  [{verdict}]")
    print(f"best-combo per-fold: {[round(f,4) for f in bfs]} std={np.std(bfs):.4f}")

    # show the L2/L5 precision-recall slide along m (with the other fixed at 1.0)
    print("\nL2 precision/recall slide (m5=1.0):")
    for m2 in MULTS:
        oof, _, _, _, _ = lgbm_oof(X, y, groups, params=TUNED, weight_fn=make_weight_fn(m2, 1.0))
        p = precision_score(y, oof, labels=[2], average="micro", zero_division=0)
        r = recall_score(y, oof, labels=[2], average="micro", zero_division=0)
        f = f1_score(y, oof, average=None, labels=[2])[0]
        print(f"  m2={m2:<4}: L2 prec={p:.4f} rec={r:.4f} F1={f:.4f}")
    print("L5 precision/recall slide (m2=1.0):")
    for m5 in MULTS:
        oof, _, _, _, _ = lgbm_oof(X, y, groups, params=TUNED, weight_fn=make_weight_fn(1.0, m5))
        p = precision_score(y, oof, labels=[5], average="micro", zero_division=0)
        r = recall_score(y, oof, labels=[5], average="micro", zero_division=0)
        f = f1_score(y, oof, average=None, labels=[5])[0]
        print(f"  m5={m5:<4}: L5 prec={p:.4f} rec={r:.4f} F1={f:.4f}")

    if d > NOISE:
        print(f"\n>>> beats baseline by > noise -> writing weighted submission")
        write_submission(bm2, bm5)
    else:
        print(f"\n>>> WASH -> not adopted. Locked model (0.7236) stays.")


def write_submission(m2, m5):
    full_params = dict(TUNED); full_params["n_estimators"] = 463
    X_test_raw, meta_test = D.load_split("test")
    assert "label" not in meta_test.columns
    X_test = build_features(X_test_raw)

    def predict_all():
        model = fit_full(X, y, params=full_params, weight_fn=make_weight_fn(m2, m5))
        return model.predict(X_test)

    p1 = predict_all(); p2 = predict_all()
    assert np.array_equal(p1, p2), "weighted refit NOT reproducible!"
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
