"""STEP A -- LightGBM hyperparameter tuning, optimizing OOF macro-F1.

Selection criterion = the StratifiedGroupKFold POOLED OOF macro-F1 (the same harness
as every prior step), NOT a single split and NOT logloss/accuracy. Inverse-frequency
sample weights are kept. Optuna TPE, seeded for reproducibility. We keep the search
space modest and, among the top trials, prefer stable ones (low per-fold std) over a
config that wins pooled OOF on one lucky fold.

Run: python step5_tuneA.py
"""
from __future__ import annotations
import numpy as np
import optuna

import har_data as D
from har_cv import SEED, N_CLASSES
from lgbm_cv import BASE_PARAMS, lgbm_oof
from step2_familyB import build_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 60
LGBM_BASELINE = {"macro": 0.7095,
                 "per_class": [0.9648, 0.8976, 0.2238, 0.6991, 0.8218, 0.6496]}

X_raw, meta = D.load_split("train")
y = meta["label"].to_numpy()
groups = meta["user"].to_numpy()
X = build_features(X_raw)


def objective(trial):
    params = dict(BASE_PARAMS)
    params.update(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
    )
    _, macro, _, fold_f1s, _ = lgbm_oof(X, y, groups, params=params)
    trial.set_user_attr("fold_std", float(np.std(fold_f1s)))
    trial.set_user_attr("fold_f1s", [float(f) for f in fold_f1s])
    return macro


def main():
    print(f"Tuning LightGBM: {N_TRIALS} trials, optimizing pooled OOF macro-F1 "
          f"(StratifiedGroupKFold). Baseline 0.7095.\n")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # --- inspect top trials for stability ---
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    print("Top 5 trials (pooled OOF macro-F1, per-fold std):")
    for t in trials[:5]:
        print(f"  trial {t.number:>3}: macro={t.value:.4f}  fold_std={t.user_attrs['fold_std']:.4f}  "
              f"folds={[round(f,3) for f in t.user_attrs['fold_f1s']]}")

    # Prefer the most stable among trials within 0.002 of the best pooled OOF.
    best_val = trials[0].value
    contenders = [t for t in trials if best_val - t.value <= 0.002]
    chosen = min(contenders, key=lambda t: t.user_attrs["fold_std"])
    print(f"\nChose trial {chosen.number}: macro={chosen.value:.4f}, "
          f"fold_std={chosen.user_attrs['fold_std']:.4f} "
          f"({'best pooled OOF' if chosen.number == trials[0].number else 'more stable, within 0.002 of best'})")

    best_params = dict(BASE_PARAMS); best_params.update(chosen.params)
    # re-run chosen to get per-class
    _, macro, per_class, fold_f1s, best_iters = lgbm_oof(X, y, groups, params=best_params)

    print("\n" + "=" * 86)
    hdr = f"{'model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))

    row("LightGBM (default, Step 3)", LGBM_BASELINE["macro"], LGBM_BASELINE["per_class"])
    row("LightGBM (tuned, Step A)", macro, per_class)

    print(f"\nL2: {LGBM_BASELINE['per_class'][2]:.4f} -> {per_class[2]:.4f} "
          f"({per_class[2]-LGBM_BASELINE['per_class'][2]:+.4f})")
    print(f"L5: {LGBM_BASELINE['per_class'][5]:.4f} -> {per_class[5]:.4f} "
          f"({per_class[5]-LGBM_BASELINE['per_class'][5]:+.4f})")
    print(f"macro: {LGBM_BASELINE['macro']:.4f} -> {macro:.4f} ({macro-LGBM_BASELINE['macro']:+.4f})")
    print(f"per-fold: {[round(f,4) for f in fold_f1s]}  std={np.std(fold_f1s):.4f} "
          f"(default std was 0.0366)")
    print(f"best_iterations: {best_iters}")
    print("\nCHOSEN PARAMS:")
    for k in ("learning_rate","num_leaves","min_child_samples","colsample_bytree",
              "subsample","reg_alpha","reg_lambda","min_split_gain"):
        print(f"  {k}: {best_params[k]}")


if __name__ == "__main__":
    main()
