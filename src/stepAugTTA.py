"""Push augmentation further: train-augmentation + TEST-TIME augmentation (TTA).

Train on real + synthetic subjects (as in the LB-0.7904 winner), then at inference predict
each file on M rotated/scaled VIEWS + the original and average -> more robust, subject-
generalizing predictions. Gateable: the same TTA is applied to validation files in OOF,
exactly mirroring test. Fair 3-partition gate vs base (0.7287) and current aug (~0.734).

FIX: augmentation seeds now use a deterministic user id (not Python's per-process-salted
hash), so results are bit-reproducible across runs.

Run: python stepAugTTA.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from lgbm_cv import BASE_PARAMS, INTERNAL_VAL_FRAC
from step2_familyB import build_features

TUNED = dict(BASE_PARAMS); TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
K_AUG, ROT, SM, SS = 2, 20.0, 0.10, 0.20      # train-augmentation (LB-validated config)
M_TTA = 6                                       # test-time views (+ original)
CURRENT_AUG_MEAN = 0.7341
OUT = "/root/dm-assignment3/submission_tta.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def uid(u):
    return int(str(u).split("_")[1])           # deterministic, process-independent


def rot_mat(rng, deg):
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-9)
    a = np.deg2rad(rng.uniform(-deg, deg))
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def transform(Xr, rng, deg, sm, ss, jit=0.01):
    mean = (Xr[..., :3] @ rot_mat(rng, deg).T) * rng.uniform(1 - sm, 1 + sm)
    std = np.abs(Xr[..., 3:] * rng.uniform(1 - ss, 1 + ss))
    return np.concatenate([mean, std], axis=-1) + rng.normal(0, jit, Xr.shape)


def aug_train(Xr, user_sub, k):
    out = Xr.copy()
    for u in np.unique(user_sub):
        m = user_sub == u
        rng = np.random.RandomState((L.SEED * 100003 + k * 9973 + uid(u)) % (2**32))
        out[m] = transform(out[m], rng, ROT, SM, SS)
    return out


def tta_views(Xr):
    """Original + M global rotated/scaled views (vectorized, deterministic)."""
    views = [Xr]
    for v in range(M_TTA):
        rng = np.random.RandomState(L.SEED + 1 + v)
        R = rot_mat(rng, ROT); a = rng.uniform(1 - SM, 1 + SM); b = rng.uniform(1 - SS, 1 + SS)
        mean = (Xr[..., :3] @ R.T) * a; std = np.abs(Xr[..., 3:] * b)
        views.append(np.concatenate([mean, std], axis=-1))
    return views


def tta_predict(model, Xr_subset):
    probs = np.zeros((len(Xr_subset), L.N_CLASSES))
    for vw in tta_views(Xr_subset):
        probs += model.predict_proba(build_features(vw))
    return probs / (M_TTA + 1)


def train_one(X_raw, X86, y, user, tr):
    sub_tr, sub_es = train_test_split(tr, test_size=INTERNAL_VAL_FRAC,
                                      stratify=y[tr], random_state=L.SEED)
    feats = [X86[sub_tr]]; labs = [y[sub_tr]]
    for k in range(K_AUG):
        feats.append(build_features(aug_train(X_raw[sub_tr], user[sub_tr], k))); labs.append(y[sub_tr])
    Xtr = np.vstack(feats); ytr = np.concatenate(labs)
    m = lgb.LGBMClassifier(**TUNED)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr),
          eval_set=[(X86[sub_es], y[sub_es])], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m


def oof(X_raw, X86, y, fid, user, cv, use_tta):
    out = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X86, y, user):
        m = train_one(X_raw, X86, y, user, tr)
        out[va] = tta_predict(m, X_raw[va]) if use_tta else m.predict_proba(X86[va])
    return out


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    parts = Dl.get_partitions(X86, y, fid, user)
    base_hmm = [p[3] for p in parts]
    print(f"base ref mean={np.mean(base_hmm):.4f}; current aug (no TTA) ~{CURRENT_AUG_MEAN}; "
          f"train-aug K{K_AUG} rot{ROT}, TTA M{M_TTA}\n")

    no_tta, with_tta, p0 = [], [], None
    for i, (s, cv, _lp, _bh) in enumerate(parts):
        o_n = oof(X_raw, X86, y, fid, user, cv, use_tta=False)
        o_t = oof(X_raw, X86, y, fid, user, cv, use_tta=True)
        sm_n = L.smooth(o_n, y, fid, user, user, cv, **L.CURRENT)
        sm_t = L.smooth(o_t, y, fid, user, user, cv, **L.CURRENT)
        no_tta.append(f1_score(y, sm_n, average="macro"))
        with_tta.append(f1_score(y, sm_t, average="macro"))
        if i == 0:
            p0 = sm_t
        print(f"  partition {s}: aug(no TTA)+HMM={no_tta[-1]:.4f}  aug+TTA+HMM={with_tta[-1]:.4f}")

    print("\n" + "=" * 70)
    print(f"{'variant':26} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate")
    print(f"{'base LGB+HMM (ref)':26} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} | ref")
    for name, ms in [("aug (no TTA) + HMM", no_tta), ("aug + TTA + HMM", with_tta)]:
        print(f"{name:26} | {np.mean(ms):7.4f} | {ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} "
              f"| {Dl.gate(ms, base_hmm)}")

    pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
    print("\naug+TTA+HMM per-class (part0): " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]}")

    print(f"\nTTA delta vs no-TTA (mean): {np.mean(with_tta)-np.mean(no_tta):+.4f}")
    if Dl.gate(with_tta, base_hmm) and np.mean(with_tta) > CURRENT_AUG_MEAN + Dl.NOISE:
        print(f">>> TTA beats current aug by > noise -> writing {OUT}")
        write_submission(X_raw, X86, y, fid, user)
    else:
        print(">>> TTA gain within noise of current aug; keep submission_aug.csv (0.7904) "
              "unless you want to try it on LB.")


def write_submission(X_raw, X86, y, fid, user):
    full = dict(TUNED); full["n_estimators"] = 463
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    T, prior = L.estimate_transition(y, fid, user)

    def fit_pred():
        feats = [X86]; labs = [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_train(X_raw, user, k))); labs.append(y)
        m = lgb.LGBMClassifier(**full)
        m.fit(np.vstack(feats), np.concatenate(labs),
              sample_weight=compute_sample_weight("balanced", np.concatenate(labs)))
        probs = tta_predict(m, Xte_raw)
        return L.decode_test(probs, fid_te, user_te, T, prior, **L.CURRENT)

    p1 = fit_pred(); p2 = fit_pred()
    assert np.array_equal(p1, p2), "not reproducible!"
    sample = pd.read_csv(SUB_TEMPLATE)
    pbi = dict(zip(fid_te.tolist(), p1.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print(f"Reproducible. Label dist: {dict(sorted(Counter(out.tolist()).items()))}\nWrote {OUT}")


if __name__ == "__main__":
    main()
