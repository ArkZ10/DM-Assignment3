"""STEP A -- HMM smoother hyperparameter tuning (Optuna, 3-partition robustness).

Sweeps: transition strength s, emission weight e (LGB prob weight vs uniform), prior
correction beta, decode (viterbi/forward-backward), transition (empirical/sticky), and
log-vs-raw probabilities. Optimizes the MEAN OOF macro-F1 across 3 fold partitions.
Adopts only if it clears the robustness gate vs current (mean +>0.0024 AND not worse on
any partition); if so, writes submission_hmmtuned.csv.

Run: python stepA_hmm_tune.py
"""
from __future__ import annotations
import numpy as np
import optuna
import temporal_lib as L

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = 50


def main():
    X, y, fid, user = L.load_train()
    groups = user
    print("Computing base OOF probabilities for 3 partitions (once)...")
    parts = L.make_partitions(X, y, groups)

    base_ms = L.eval_macro(parts, y, fid, user, groups, **L.CURRENT)
    print(f"current smoother across partitions: mean={base_ms.mean():.4f} "
          f"{[round(m,4) for m in base_ms]}\n")

    def objective(trial):
        cfg = dict(
            s=trial.suggest_float("s", 0.1, 2.0),
            e=trial.suggest_float("e", 0.3, 2.0),
            beta=trial.suggest_float("beta", 0.0, 1.5),
            decode=trial.suggest_categorical("decode", ["viterbi", "fb"]),
            trans=trial.suggest_categorical("trans", ["emp", "sticky"]),
            prob_mode=trial.suggest_categorical("prob_mode", ["log", "raw"]),
        )
        ms = L.eval_macro(parts, y, fid, user, groups, **cfg)
        trial.set_user_attr("ms", ms.tolist())
        return ms.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=L.SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    print("Top 5 configs (mean OOF macro across partitions):")
    for t in trials[:5]:
        print(f"  mean={t.value:.4f} ms={[round(x,4) for x in t.user_attrs['ms']]}  {t.params}")

    best = trials[0]
    best_cfg = {k: best.params[k] for k in ("s", "e", "beta", "decode", "trans", "prob_mode")}
    cand_ms = np.array(best.user_attrs["ms"])
    win = L.robust_gate(cand_ms, base_ms)

    print("\n" + "=" * 60)
    print(f"current : mean={base_ms.mean():.4f} {[round(m,4) for m in base_ms]}")
    print(f"best    : mean={cand_ms.mean():.4f} {[round(m,4) for m in cand_ms]}")
    print(f"delta mean: {cand_ms.mean()-base_ms.mean():+.4f}  (noise {L.NOISE})")
    print(f"best config: {best_cfg}")
    print(f"ROBUST WIN (mean +>{L.NOISE} AND not worse on any partition): {win}")

    if win:
        print("\n>>> robust win -> writing submission_hmmtuned.csv")
        L.write_submission(X, y, fid, user, best_cfg,
                           "/root/dm-assignment3/submission_hmmtuned.csv")
    else:
        print("\n>>> not a robust win -> keep submission_temporal.csv.")


if __name__ == "__main__":
    main()
