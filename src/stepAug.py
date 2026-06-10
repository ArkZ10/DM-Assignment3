"""Idea B -- cross-subject augmentation: manufacture synthetic subjects, train LGB+HMM.

For each TRAINING user we create K augmented copies via subject-level transforms applied
to the raw 300x6 sequence: a random 3D rotation of the gravity/orientation channels
(mean_x/y/z -> device-placement variation), magnitude scaling of motion (std_* ->
intensity variation), and small jitter. Params are per-user (one coherent "new subject"
per copy). We recompute the 86 features on the augmented sequences and train LightGBM on
real + synthetic subjects, then HMM. Train-only (val/test never augmented -> no leakage).

Fair 3-partition gate vs LGB+HMM 0.7287. Also reports the augmented model blended with
the base LGB. Ref: Diverse Intra/Inter-Domain Style Fusion (arXiv 2406.04609).

Run: python stepAug.py
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from lgbm_cv import BASE_PARAMS, INTERNAL_VAL_FRAC
from step2_familyB import build_features
import json

TUNED = dict(BASE_PARAMS); TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
K_AUG = 2                      # augmented copies per real user
ROT_DEG, SCALE_M, SCALE_S, JIT = 20.0, 0.10, 0.20, 0.01


def rotation_matrix(rng):
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-9)
    ang = np.deg2rad(rng.uniform(-ROT_DEG, ROT_DEG))
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)


def augment_user_raw(Xu, rng):
    R = rotation_matrix(rng)
    sm = rng.uniform(1 - SCALE_M, 1 + SCALE_M)
    ss = rng.uniform(1 - SCALE_S, 1 + SCALE_S)
    mean = (Xu[..., :3] @ R.T) * sm
    std = np.abs(Xu[..., 3:] * ss)
    out = np.concatenate([mean, std], axis=-1)
    return out + rng.normal(0, JIT, out.shape)


def augment_subset(Xraw_sub, user_sub, seed, k):
    out = Xraw_sub.copy()
    for u in np.unique(user_sub):
        m = user_sub == u
        rng = np.random.RandomState((seed * 100003 + k * 9973 + (hash(u) % 100000)) % (2**32))
        out[m] = augment_user_raw(out[m], rng)
    return out


def train_predict_aug(X_raw, X86, y, user, tr, va, K, seed=L.SEED):
    sub_tr, sub_es = train_test_split(tr, test_size=INTERNAL_VAL_FRAC,
                                      stratify=y[tr], random_state=seed)
    feats = [X86[sub_tr]]; labs = [y[sub_tr]]
    for k in range(K):
        Xaug = augment_subset(X_raw[sub_tr], user[sub_tr], seed, k)
        feats.append(build_features(Xaug)); labs.append(y[sub_tr])
    Xtr = np.vstack(feats); ytr = np.concatenate(labs)
    sw = compute_sample_weight("balanced", ytr)
    m = lgb.LGBMClassifier(**TUNED)
    m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(X86[sub_es], y[sub_es])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(X86[va])


def oof_aug(X_raw, X86, y, user, cv, K):
    groups = user
    oof = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X86, y, groups):
        oof[va] = train_predict_aug(X_raw, X86, y, user, tr, va, K)
    return oof


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    groups = user

    parts = Dl.get_partitions(X86, y, fid, user)        # base lgb_probs + hmm ref
    base_hmm = [p[3] for p in parts]
    print(f"base LGB+HMM ref: {[round(h,4) for h in base_hmm]} mean={np.mean(base_hmm):.4f}")
    print(f"augmenting {K_AUG} synthetic subjects/user (rot {ROT_DEG}deg, "
          f"scale m+-{SCALE_M}/s+-{SCALE_S}, jit {JIT})\n")

    aug_hmm, blend_hmm, p0 = [], [], None
    for i, (s, cv, lgb_p, _bh) in enumerate(parts):
        probs = oof_aug(X_raw, X86, y, user, cv, K_AUG)
        sm = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        aug_hmm.append(f1_score(y, sm, average="macro"))
        smb = L.smooth(0.5 * probs + 0.5 * lgb_p, y, fid, user, groups, cv, **L.CURRENT)
        blend_hmm.append(f1_score(y, smb, average="macro"))
        if i == 0:
            p0 = sm
        print(f"  partition {s}: aug+HMM={aug_hmm[-1]:.4f}  aug+LGB blend+HMM={blend_hmm[-1]:.4f}  "
              f"(base {_bh:.4f})")

    print("\n" + "=" * 80)
    hdr = f"{'variant':28} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate"
    print(hdr); print("-" * len(hdr))
    print(f"{'base LGB + HMM (ref)':28} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} |  ref")
    for name, ms in [("augmented LGB + HMM", aug_hmm), ("0.5 aug + 0.5 LGB + HMM", blend_hmm)]:
        print(f"{name:28} | {np.mean(ms):7.4f} | {ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} "
              f"| {Dl.gate(ms, base_hmm)}")

    pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
    print("\naugmented+HMM per-class (part0): " +
          " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]} (base ~78)")

    best = max([("aug", aug_hmm), ("blend", blend_hmm)], key=lambda r: np.mean(r[1]))
    print(f"\nbest: {best[0]} mean={np.mean(best[1]):.4f}  "
          f"ROBUST WIN={Dl.gate(best[1], base_hmm)}")
    if not Dl.gate(best[1], base_hmm):
        print(">>> no robust win -> keep submission_temporal.csv (0.7867).")


if __name__ == "__main__":
    main()
