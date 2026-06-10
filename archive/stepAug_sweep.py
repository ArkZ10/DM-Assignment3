"""Push the winning augmentation harder: sweep K (synthetic subjects/user) and transform
strength, selected on the fair 3-partition gate. Current best: K=2, rot20, sm.1, ss.2 ->
OOF 0.7341 / LB 0.7904. If a stronger config beats it by > noise on all 3 partitions,
generate submission_aug_v2.csv (full-data refit, reproducibility-checked).

Run: python stepAug_sweep.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from lgbm_cv import BASE_PARAMS, INTERNAL_VAL_FRAC
from step2_familyB import build_features

TUNED = dict(BASE_PARAMS); TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
CURRENT_AUG_MEAN = 0.7341       # K=2 rot20 sm.1 ss.2 (LB 0.7904)
OUT = "/root/dm-assignment3/submission_aug_v2.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"

# (K, rot_deg, scale_mean, scale_std)
CONFIGS = [
    ("K2 rot20 sm.10 ss.20 (current)", 2, 20, 0.10, 0.20),
    ("K4 rot20 sm.10 ss.20",           4, 20, 0.10, 0.20),
    ("K3 rot30 sm.15 ss.30",           3, 30, 0.15, 0.30),
    ("K4 rot30 sm.15 ss.30",           4, 30, 0.15, 0.30),
]


def rot_mat(rng, deg):
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-9)
    a = np.deg2rad(rng.uniform(-deg, deg))
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * (Kx @ Kx)


def aug_raw(Xu, rng, deg, sm, ss):
    R = rot_mat(rng, deg)
    mean = (Xu[..., :3] @ R.T) * rng.uniform(1 - sm, 1 + sm)
    std = np.abs(Xu[..., 3:] * rng.uniform(1 - ss, 1 + ss))
    return np.concatenate([mean, std], axis=-1) + rng.normal(0, 0.01, Xu.shape)


def aug_subset(Xr, user_sub, seed, k, deg, sm, ss):
    out = Xr.copy()
    for u in np.unique(user_sub):
        m = user_sub == u
        rng = np.random.RandomState((seed * 100003 + k * 9973 + (hash(u) % 100000)) % (2**32))
        out[m] = aug_raw(out[m], rng, deg, sm, ss)
    return out


def oof_aug(X_raw, X86, y, user, cv, K, deg, sm, ss):
    oof = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X86, y, user):
        sub_tr, sub_es = train_test_split(tr, test_size=INTERNAL_VAL_FRAC,
                                          stratify=y[tr], random_state=L.SEED)
        feats = [X86[sub_tr]]; labs = [y[sub_tr]]
        for k in range(K):
            feats.append(build_features(aug_subset(X_raw[sub_tr], user[sub_tr], L.SEED, k, deg, sm, ss)))
            labs.append(y[sub_tr])
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        m = lgb.LGBMClassifier(**TUNED)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr),
              eval_set=[(X86[sub_es], y[sub_es])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X86[va])
    return oof


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    parts = Dl.get_partitions(X86, y, fid, user)
    base_hmm = [p[3] for p in parts]
    print(f"base ref {[round(h,4) for h in base_hmm]} mean={np.mean(base_hmm):.4f}; "
          f"current aug LB-best mean={CURRENT_AUG_MEAN}\n")

    rows = []
    for name, K, deg, sm, ss in CONFIGS:
        ms = []
        for s, cv, _lp, _bh in parts:
            probs = oof_aug(X_raw, X86, y, user, cv, K, deg, sm, ss)
            sm_pred = L.smooth(probs, y, fid, user, user, cv, **L.CURRENT)
            ms.append(f1_score(y, sm_pred, average="macro"))
        rows.append((name, K, deg, sm, ss, ms))
        print(f"{name:32} mean={np.mean(ms):.4f} {[round(m,4) for m in ms]} "
              f"gate={Dl.gate(ms, base_hmm)}")

    print("\n" + "=" * 78)
    hdr = f"{'config':32} | {'mean':>7} | gate-vs-base"
    print(hdr); print("-" * len(hdr))
    print(f"{'base LGB+HMM':32} | {np.mean(base_hmm):7.4f} | ref")
    for name, K, deg, sm, ss, ms in rows:
        print(f"{name:32} | {np.mean(ms):7.4f} | {Dl.gate(ms, base_hmm)}")

    best = max(rows, key=lambda r: np.mean(r[5]))
    bmean = np.mean(best[5])
    print(f"\nbest config: {best[0]} mean={bmean:.4f} (current aug {CURRENT_AUG_MEAN})")
    if Dl.gate(best[5], base_hmm) and bmean > CURRENT_AUG_MEAN + Dl.NOISE:
        print(f">>> beats current aug by > noise -> generating {OUT}")
        write_submission(X_raw, X86, y, fid, user, best[1], best[2], best[3], best[4])
    elif best[0].startswith("K2 rot20"):
        print(">>> current config remains best; keep submission_aug.csv (LB 0.7904).")
    else:
        print(f">>> best beats base but not current-aug by >noise; marginal. "
              f"submission_aug.csv (0.7904) stays unless you want to try {best[0]} on LB.")


def write_submission(X_raw, X86, y, fid, user, K, deg, sm, ss):
    full = dict(TUNED); full["n_estimators"] = 463
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    def fit_pred():
        feats = [X86]; labs = [y]
        for k in range(K):
            feats.append(build_features(aug_subset(X_raw, user, L.SEED, k, deg, sm, ss)))
            labs.append(y)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        m = lgb.LGBMClassifier(**full)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        return L.decode_test(m.predict_proba(X86_te), fid_te, user_te, T, prior, **L.CURRENT)

    p1 = fit_pred(); p2 = fit_pred()
    assert np.array_equal(p1, p2), "not reproducible!"
    sample = pd.read_csv(SUB_TEMPLATE)
    pred_by_id = dict(zip(fid_te.tolist(), p1.tolist()))
    out = np.array([pred_by_id[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pred_by_id) == set(sample["Id"].tolist()) and len(sample) == 6849
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(OUT, index=False)
    print(f"Reproducible. Label dist: {dict(sorted(Counter(out.tolist()).items()))}\nWrote {OUT}")


if __name__ == "__main__":
    main()
