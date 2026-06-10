"""Trial 1 of 2: apply gate-validated per-class FB calibration to the proven
0.8061 pseudo-aug ensemble (gen_pseudo_aug.py).

stepB_v2.py validated under the FAIR decoder_lib 3-partition gate:
  FB decode + class_mult=[1, 1.221, 1.492, 1, 1, 1.105]  vs  Viterbi (L.CURRENT)
  -> +0.0046 mean, robust win on all 3 partitions (incl. L2 bottleneck gain).

This is a pure DECODE-time change (orthogonal to the model/data that produced the
probabilities) -- identical 5+7 member ensemble, identical seeds/data as
gen_pseudo_aug.py (LB 0.8061), only the final argmax step changes:
  Viterbi(probs)              -> submission_pseudo_aug.csv   (0.8061, unchanged)
  FB(probs) * class_mult       -> submission_pseudo_calib.csv (NEW, this trial)

Run: python gen_pseudo_calib.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

import har_data as D
import temporal_lib as L
from step2_familyB import build_features
from stepAugTTA import TUNED, K_AUG, ROT, SM, SS, transform, uid

THRESH = 0.82
K_AUG_PSEUDO = 1
N_PASS1 = 5
N_PASS2 = 7
PSEUDO_SEED = 777777
CLASS_MULT = np.load("/root/dm-assignment3/cache/class_mult.npy")
CFG_FB = dict(s=L.CURRENT["s"], e=L.CURRENT["e"], beta=L.CURRENT["beta"],
               decode="fb", trans=L.CURRENT["trans"], prob_mode=L.CURRENT["prob_mode"])
OUT_VIT = "/root/dm-assignment3/submission_pseudo_aug_check.csv"
OUT_CAL = "/root/dm-assignment3/submission_pseudo_calib.csv"
SUB_TEMPLATE = "/root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv"


def aug_member(X_raw, user, k, member, seed_base=L.SEED):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (seed_base * 100003 + member * 777767 + k * 9973 + uid(u)) % (2**32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def run_ensemble(X_raw, X86, y, user, X86_te, n_members, seed_offset,
                 Xte_raw_pseudo=None, X86_pseudo=None, y_pseudo=None, user_pseudo=None):
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(n_members):
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_member(X_raw, user, k, e)))
            labs.append(y)
        if X86_pseudo is not None:
            feats.append(X86_pseudo); labs.append(y_pseudo)
            for k in range(K_AUG_PSEUDO):
                feats.append(build_features(aug_member(Xte_raw_pseudo, user_pseudo,
                                                       k, e, seed_base=PSEUDO_SEED)))
                labs.append(y_pseudo)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        p = dict(full); p["random_state"] = L.SEED + seed_offset + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"  [offset={seed_offset}] member {e}/{n_members-1} done")
    return proba / n_members


def full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior):
    print(f"Pass 1 ({N_PASS1} members, original data only)...")
    pr1 = run_ensemble(X_raw, X86, y, user, X86_te, N_PASS1, seed_offset=0)

    mask = pr1.max(axis=1) > THRESH
    y_pseudo = pr1.argmax(axis=1)[mask]
    X86_pseudo = X86_te[mask]
    Xte_raw_pseudo = Xte_raw[mask]
    user_pseudo = user_te[mask]
    print(f"\nPseudo-labeled {mask.sum()}/{len(X86_te)} "
          f"({mask.mean()*100:.1f}%, thresh={THRESH}), "
          f"mean conf={pr1.max(1)[mask].mean():.4f}")
    print(f"Class dist: {dict(sorted(Counter(y_pseudo.tolist()).items()))}")

    rows_real = len(X86) * (1 + K_AUG)
    rows_pseudo = len(X86_pseudo) * (1 + K_AUG_PSEUDO)
    print(f"\nPass 2 ({N_PASS2} members, {rows_real} real + {rows_pseudo} pseudo rows)...")
    pr2 = run_ensemble(X_raw, X86, y, user, X86_te, N_PASS2, seed_offset=500,
                       Xte_raw_pseudo=Xte_raw_pseudo, X86_pseudo=X86_pseudo,
                       y_pseudo=y_pseudo, user_pseudo=user_pseudo)
    return pr2


def decode_fb_mult(probs, fid, user, T, prior, cfg, mult):
    log_T = cfg["s"] * np.log(T + L.EPS); log_prior = cfg["s"] * np.log(prior + L.EPS)
    pred = np.full(len(fid), -1, dtype=int)
    for u in np.unique(user):
        idx = np.where(user == u)[0]; idx = idx[np.argsort(fid[idx])]
        em = L._emission(probs[idx], prior, cfg["e"], cfg["beta"], cfg["prob_mode"])
        post = np.exp(L.forward_backward(em, log_T, log_prior))
        pred[idx] = (post * mult).argmax(1)
    return pred


def write_csv(pred, fid_te, sample, out_path):
    pbi = dict(zip(fid_te.tolist(), pred.tolist()))
    out = np.array([pbi[i] for i in sample["Id"].to_numpy()], dtype=int)
    assert set(pbi) == set(sample["Id"].tolist()) and len(sample) == 6849
    assert set(np.unique(out)).issubset(set(range(6)))
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "Label": out}).to_csv(out_path, index=False)
    print("Label dist:", {c: Counter(out.tolist()).get(c, 0) for c in range(L.N_CLASSES)})
    print(f"Wrote {out_path}")


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    Xte_raw, meta_te = D.load_split("test")
    fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
    X86_te = build_features(Xte_raw)
    T, prior = L.estimate_transition(y, fid, user)

    print("=== Trial 1: FB + class_mult calibration on the 0.8061 pseudo-aug ensemble ===")
    print(f"class_mult = {CLASS_MULT}\n")
    pr2_a = full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior)

    print("\nReproducibility check (re-running pass1+pass2)...")
    pr2_b = full_run(X_raw, X86, y, user, fid, Xte_raw, X86_te, fid_te, user_te, T, prior)
    print(f"pr2 reproducible: {np.array_equal(pr2_a, pr2_b)}")
    assert np.array_equal(pr2_a, pr2_b), "NOT reproducible!"

    pred_vit = L.decode_test(pr2_a, fid_te, user_te, T, prior, **L.CURRENT)
    pred_cal = decode_fb_mult(pr2_a, fid_te, user_te, T, prior, CFG_FB, CLASS_MULT)

    n_diff = (pred_vit != pred_cal).sum()
    print(f"\nViterbi vs FB+mult: {n_diff}/{len(pred_vit)} predictions differ "
          f"({n_diff/len(pred_vit)*100:.1f}%)")
    print("Viterbi class dist:", dict(sorted(Counter(pred_vit.tolist()).items())))
    print("FB+mult  class dist:", dict(sorted(Counter(pred_cal.tolist()).items())))

    sample = pd.read_csv(SUB_TEMPLATE)
    print("\n--- sanity (should match submission_pseudo_aug.csv exactly) ---")
    write_csv(pred_vit, fid_te, sample, OUT_VIT)
    print("\n--- new candidate ---")
    write_csv(pred_cal, fid_te, sample, OUT_CAL)


if __name__ == "__main__":
    main()
