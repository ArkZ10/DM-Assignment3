"""Step 1: validation harness sanity check.

Builds ONLY the 12 baseline features (per-file mean + std of each of the 6 sensor
columns), then:
  (1) cross-checks against the audit using plain GroupKFold -> must reproduce 0.5476
      exactly, proving the feature + scoring code matches the audit.
  (2) reports the StratifiedGroupKFold number, which is the harness used from Step 2 on.

No feature engineering yet. Run: python step1_harness.py
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import har_data as D
from har_cv import SEED, evaluate_oof

np.random.seed(SEED)

AUDIT_GROUPKFOLD_MACRO = 0.5476  # value reported by the standalone audit


def baseline_12_features(X_raw: np.ndarray) -> np.ndarray:
    """Collapse each (300, 6) file to 12 features: per-column mean and std.

    ddof=1 to match the audit (pandas .std() default), so the GroupKFold
    cross-check reproduces the audit number exactly.
    """
    mean = X_raw.mean(axis=1)               # (n, 6)
    std = X_raw.std(axis=1, ddof=1)         # (n, 6)
    return np.concatenate([mean, std], axis=1)  # (n, 12)


def lr_factory():
    # lbfgs is deterministic; random_state pinned for belt-and-suspenders repro.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )


def main():
    X_raw, meta = D.load_split("train")
    X = baseline_12_features(X_raw)
    y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    print(f"design matrix: X={X.shape}, users={len(set(groups))}, classes={sorted(set(y))}")

    print("\n[cross-check] plain GroupKFold (must match audit 0.5476):")
    _, macro_gkf, _, _ = evaluate_oof(X, y, groups, lr_factory, stratified=False)

    print("\n[harness] StratifiedGroupKFold (used from Step 2 onward):")
    _, macro_sgkf, per_class_sgkf, _ = evaluate_oof(X, y, groups, lr_factory, stratified=True)

    print("\n" + "=" * 60)
    delta = macro_gkf - AUDIT_GROUPKFOLD_MACRO
    ok = abs(delta) < 5e-4
    print(f"GroupKFold cross-check : {macro_gkf:.4f}  "
          f"(audit {AUDIT_GROUPKFOLD_MACRO:.4f}, delta {delta:+.4f})  "
          f"{'OK -- harness matches audit' if ok else 'MISMATCH -- STOP'}")
    print(f"StratifiedGroupKFold   : {macro_sgkf:.4f}  <-- baseline for the additive table")
    if not ok:
        raise SystemExit("Harness does NOT reproduce the audit. Stopping per spec.")


if __name__ == "__main__":
    main()
