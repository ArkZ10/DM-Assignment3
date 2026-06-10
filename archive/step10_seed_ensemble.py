"""STEP A -- seed ensemble of the tuned LightGBM (cheapest variance reduction).

Rerun the EXACT tuned config with 5 different random_state seeds, average the OOF
class-probability vectors (soft vote), argmax. Same StratifiedGroupKFold, same internal
split, inverse-freq weights. Adopt only if it beats 0.7236 by > noise floor (0.0024);
if so, refit the 5-seed ensemble on all 11,020 train files and write submission.csv.

Run: python step10_seed_ensemble.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof_proba, fit_full
from step2_familyB import build_features

NOISE = 0.0024
BASE = {"macro": 0.7236, "per_class": [0.9645, 0.8922, 0.2773, 0.7136, 0.8561, 0.6378]}
SEEDS = [42, 1, 2, 3, 4]
N_ESTIMATORS_FULL = 463  # median CV best_iter from the locked config (Step D)
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"
OUT = "/root/dm-assignment3/submission_seedensemble.csv"

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))


def params_with_seed(seed):
    p = dict(TUNED)
    p["random_state"] = seed
    return p


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    X = build_features(X_raw)
    print(f"STEP A -- seed ensemble. Baseline 0.7236, noise +/-{NOISE}. Seeds={SEEDS}\n")

    proba_sum = np.zeros((len(y), N_CLASSES))
    print("Per-seed OOF macro-F1:")
    for s in SEEDS:
        oof_p = lgbm_oof_proba(X, y, groups, params=params_with_seed(s))
        proba_sum += oof_p
        m = f1_score(y, oof_p.argmax(1), average="macro")
        print(f"  seed {s:>3}: macro-F1 = {m:.4f}")

    ens_pred = (proba_sum / len(SEEDS)).argmax(1)
    macro = f1_score(y, ens_pred, average="macro")
    per_class = f1_score(y, ens_pred, average=None, labels=list(range(N_CLASSES)))

    print("\n" + "=" * 86)
    hdr = f"{'model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))

    row("single seed (locked, Step D)", BASE["macro"], BASE["per_class"])
    row("5-seed soft-vote ensemble", macro, per_class)

    d = macro - BASE["macro"]
    verdict = "WASH (<= noise floor)" if d <= NOISE else "WIN (> noise floor)"
    print(f"\nL2: {BASE['per_class'][2]:.4f} -> {per_class[2]:.4f} ({per_class[2]-BASE['per_class'][2]:+.4f})")
    print(f"L5: {BASE['per_class'][5]:.4f} -> {per_class[5]:.4f} ({per_class[5]-BASE['per_class'][5]:+.4f})")
    print(f"macro: {BASE['macro']:.4f} -> {macro:.4f} ({d:+.4f})  [{verdict}]")

    cm = confusion_matrix(y, ens_pred, labels=list(range(N_CLASSES)))
    print("\nCONFUSION MATRIX (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")

    if d > NOISE:
        print(f"\n>>> beats baseline by > noise floor -> writing seed-ensemble submission")
        write_submission(X, y)
    else:
        print(f"\n>>> WASH -> not adopted, no submission written. Locked model stays.")


def write_submission(X_train, y_train):
    full_params = dict(TUNED); full_params["n_estimators"] = N_ESTIMATORS_FULL
    X_test_raw, meta_test = D.load_split("test")
    assert "label" not in meta_test.columns
    X_test = build_features(X_test_raw)

    def predict_all():
        proba = np.zeros((len(X_test), N_CLASSES))
        for s in SEEDS:
            p = dict(full_params); p["random_state"] = s
            model = fit_full(X_train, y_train, params=p, use_weights=True)
            proba += model.predict_proba(X_test)
        return proba.argmax(1)

    p1 = predict_all(); p2 = predict_all()
    assert np.array_equal(p1, p2), "seed-ensemble refit NOT reproducible!"
    print("Reproducibility: two refits -> identical predictions [OK]")

    ids = meta_test["file_id"].to_numpy()
    pred_by_id = dict(zip(ids.tolist(), p1.tolist()))
    sample = pd.read_csv(SUB_TEMPLATE)
    out_labels = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist())
    assert len(sample) == 6849 and set(np.unique(out_labels)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out_labels}).to_csv(OUT, index=False)
    dist = Counter(out_labels.tolist())
    print(f"Label distribution: {{{', '.join(f'{c}:{dist.get(c,0)}' for c in range(N_CLASSES))}}}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
