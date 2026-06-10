"""Safe improvement attempt: feed a BLENDED emission into the robust temporal smoother.

The LGB+XGB+CAT blend was a WASH on base argmax (Step B), but blending usually improves
probability CALIBRATION, and the HMM/Viterbi decode consumes probabilities -- so a better-
calibrated emission can help the smoother even when argmax accuracy is unchanged. We test
whether smoothing a blended emission beats smoothing the LGB-only emission, using the
robust smoother config validated in Step 14 (Viterbi, empirical transition, beta=0.5, s=0.7).
Low overfit risk: no new hyperparameters, same smoother, blend weights reused from Step B.

Run: python step15_blend_emission.py
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

import har_data as D
from har_cv import N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof_proba
from step2_familyB import build_features
from step11_blend import xgb_oof_proba, cat_oof_proba   # parameterized, leak-free
from step13_temporal import smooth_oof

TUNED = dict(BASE_PARAMS)
TUNED.update(json.load(open("/root/dm-assignment3/tuned_params.json")))
BLEND_W = (0.45, 0.35, 0.20)     # grid-best LGB/XGB/CAT weights from Step B
BETA, S = 0.5, 0.7               # robust smoother config from Step 14

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
groups = user
X = build_features(X_raw)


def per_class(pred):
    return f1_score(y, pred, average=None, labels=list(range(N_CLASSES)))


def report(name, pred):
    m = f1_score(y, pred, average="macro")
    pc = per_class(pred)
    print(f"{name:34} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))
    return m, pc


def main():
    print("Blended-emission smoothing test (smoother: Viterbi, emp, beta=0.5, s=0.7)\n")
    print("Computing OOF probabilities: LGB, XGB, CAT (same folds)...")
    p_lgb = lgbm_oof_proba(X, y, groups, params=TUNED)
    p_xgb, _ = xgb_oof_proba(X, y, groups)
    p_cat, _ = cat_oof_proba(X, y, groups)
    p_blend = BLEND_W[0] * p_lgb + BLEND_W[1] * p_xgb + BLEND_W[2] * p_cat

    print("\n" + "=" * 86)
    hdr = f"{'emission / processing':34} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    report("LGB argmax (base, locked)", p_lgb.argmax(1))
    report("blend argmax (Step B, wash)", p_blend.argmax(1))
    sm_lgb = smooth_oof(p_lgb, y, fid, user, groups, S, BETA)
    m_lgb, _ = report("LGB emission + smoother (SUBMITTED)", sm_lgb)
    sm_blend = smooth_oof(p_blend, y, fid, user, groups, S, BETA)
    m_blend, _ = report("blend emission + smoother", sm_blend)

    d = m_blend - m_lgb
    print(f"\nblend-emission smoothing vs LGB-emission smoothing: {d:+.4f}")
    for c in (2, 5):
        p = precision_score(y, sm_blend, labels=[c], average="micro", zero_division=0)
        r = recall_score(y, sm_blend, labels=[c], average="micro", zero_division=0)
        print(f"  blend+smooth L{c}: precision={p:.4f} recall={r:.4f}")
    if d > 0.0024:
        print(">>> blended emission helps beyond single-partition noise -> worth validating "
              "across partitions before adopting.")
    else:
        print(">>> within noise -> no safe gain from blended emission; keep LGB-only smoother.")


if __name__ == "__main__":
    main()
