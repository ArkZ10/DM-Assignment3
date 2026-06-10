"""New architecture: SEGMENT / run-level classification (not window-level).

Activities come in long runs (89% stickiness). Instead of classifying each file, we:
  1. segment each user's sequence into runs of the HMM-predicted label (leak-free: OOF
     HMM labels cover every file),
  2. build run-level features (mean/std of the 86 features over the run + run length),
  3. train an LGB to classify whole SEGMENTS (label = majority true label of the run),
  4. reassign the segment's predicted label to all its files.
Run-averaging denoises borderline classes (esp. L2 runs vs L1), and exploits run
structure beyond the HMM's first-order transitions -- all at low capacity (no overfit).

Fair 3-partition gate vs the HMM input. Run: python stepSegment.py
"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features

SEG_PARAMS = dict(objective="multiclass", num_class=L.N_CLASSES, n_estimators=400,
                  learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=L.SEED, deterministic=True, force_row_wise=True,
                  n_jobs=1, verbose=-1)


def make_segments(hmm_lab, fid, user):
    """Return seg_id per file and list of segments (each = sorted file indices in a run)."""
    seg_id = np.full(len(hmm_lab), -1, dtype=int)
    segs = []
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        order = idx[np.argsort(fid[idx])]
        lab = hmm_lab[order]
        start = 0
        for i in range(1, len(order) + 1):
            if i == len(order) or lab[i] != lab[i - 1]:
                run = order[start:i]
                seg_id[run] = len(segs)
                segs.append(run)
                start = i
    return seg_id, segs


def seg_features(segs, X86):
    F = np.zeros((len(segs), 2 * X86.shape[1] + 1))
    for j, run in enumerate(segs):
        x = X86[run]
        F[j, :X86.shape[1]] = x.mean(0)
        F[j, X86.shape[1]:2 * X86.shape[1]] = x.std(0) if len(run) > 1 else 0.0
        F[j, -1] = len(run)
    return F


def main():
    X_raw, meta = L.D.load_split("train")
    X86 = build_features(X_raw)
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    parts = Dl.get_partitions(X86, y, fid, user)
    base_hmm = [p[3] for p in parts]
    print(f"base ref mean={np.mean(base_hmm):.4f}\n")

    hmm_in, seg_out, p0 = [], [], None
    for s, cv, lgb_p, _bh in parts:
        # per-file HMM OOF labels (leak-free) on this partition
        hmm_lab = L.smooth(lgb_p, y, fid, user, user, cv, **L.CURRENT)
        hmm_in.append(f1_score(y, hmm_lab, average="macro"))
        seg_id, segs = make_segments(hmm_lab, fid, user)
        SF = seg_features(segs, X86)
        seg_user = np.array([user[run[0]] for run in segs])
        seg_ytrue = np.array([np.bincount(y[run], minlength=L.N_CLASSES).argmax() for run in segs])

        refined = hmm_lab.copy()
        for tr, va in cv.split(X86, y, user):
            tr_users = set(user[tr]); va_users = set(user[va])
            tr_seg = np.array([u in tr_users for u in seg_user])
            va_seg = np.array([u in va_users for u in seg_user])
            m = lgb.LGBMClassifier(**SEG_PARAMS)
            m.fit(SF[tr_seg], seg_ytrue[tr_seg],
                  sample_weight=compute_sample_weight("balanced", seg_ytrue[tr_seg]))
            pred_seg = m.predict(SF[va_seg])
            for sidx, pl in zip(np.where(va_seg)[0], pred_seg):
                refined[segs[sidx]] = pl
        seg_out.append(f1_score(y, refined, average="macro"))
        if p0 is None:
            p0 = refined
        print(f"  partition {s}: HMM-in={hmm_in[-1]:.4f}  segment-refined={seg_out[-1]:.4f}  "
              f"({len(segs)} segments)")

    print("\n" + "=" * 70)
    print(f"{'variant':26} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate")
    print(f"{'base LGB+HMM (ref)':26} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} | ref")
    for name, ms in [("HMM input (re-check)", hmm_in), ("segment-refined", seg_out)]:
        print(f"{name:26} | {np.mean(ms):7.4f} | {ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} "
              f"| {Dl.gate(ms, base_hmm)}")

    pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
    print("\nsegment-refined per-class (part0): " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]}")
    print(f"\nsegment delta vs HMM-input (mean): {np.mean(seg_out)-np.mean(hmm_in):+.4f}")
    print(f"ROBUST WIN vs base: {Dl.gate(seg_out, base_hmm)}")


if __name__ == "__main__":
    main()
