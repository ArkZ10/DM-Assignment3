"""Generate all report figures.

Sections:
  1. Baseline & problem context        -> fig01, fig02
  2. Experimental pipeline              -> fig03..fig07
  3. Baseline vs top-3 candidates       -> fig08..fig10

All figures are saved as PNG to ./figures/. Uses cached OOF probs
(decoder_lib.get_partitions) and the gate-validated class_mult (cache/class_mult.npy)
where possible to avoid retraining; only fig03/04 (one LGB fit) and fig06 (5-member
pass-1 ensemble, reused exactly from gen_pseudo_aug.py's pass-1 seeds) train models.

Run: python make_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features
from lgbm_cv import INTERNAL_VAL_FRAC
from stepAugTTA import TUNED, K_AUG, ROT, SM, SS, transform, uid

OUTDIR = "/root/dm-assignment3/figures"
os.makedirs(OUTDIR, exist_ok=True)
SENSOR_COLS = D.SENSOR_COLS
CLASS_COLORS = plt.cm.tab10(np.arange(6))


def build_feature_names():
    names = [f"{c}_mean" for c in SENSOR_COLS] + [f"{c}_std" for c in SENSOR_COLS]
    stats = ["p10", "p25", "p50", "p75", "p90", "iqr", "max"]
    for c in SENSOR_COLS + ["mag"]:
        names += [f"{c}_{s}" for s in stats]
    names += ["mag_cross_rate", "mag_active_med", "mag_active_mean",
              "mag_longest_active", "mag_longest_inactive",
              "mag_ac1", "mag_ac2", "mag_ac3", "mag_ac4", "mag_ac5",
              "mag_peak_count", "mag_argmax_norm", "mag_first_half_frac"]
    for c in ["std_x", "std_y", "std_z"]:
        names += [f"{c}_cr", f"{c}_af", f"{c}_ac1", f"{c}_lar"]
    assert len(names) == 86
    return names


# ---------------------------------------------------------------- Section 1
def fig01_class_distribution(meta):
    counts = meta.groupby("file_id")["label"].first().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"L{c}" for c in counts.index], counts.values, color=CLASS_COLORS)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 50, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Activity label"); ax.set_ylabel("# train files")
    ratio = counts[1] / counts[4]
    ax.set_title(f"Train label distribution (n={counts.sum()} files, 60 users)\n"
                  f"imbalance ratio L1:L4 = {counts[1]}:{counts[4]} ≈ {ratio:.0f}:1")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig01_class_distribution.png", dpi=150)
    plt.close(fig)
    print("  fig01_class_distribution.png")


def fig02_data_overview(X_raw, meta, X_raw_te, meta_te):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7], hspace=0.45, wspace=0.3)
    for c in range(6):
        idx = meta.index[meta.label == c][0]
        ax = fig.add_subplot(gs[c // 3, c % 3])
        x = X_raw[idx]
        for j, col in enumerate(SENSOR_COLS):
            ax.plot(x[:, j], lw=1, label=col)
        ax.set_title(f"L{c} example (file_id={meta.loc[idx, 'file_id']})", fontsize=10)
        ax.set_xlabel("t (s)", fontsize=8)
        if c == 0:
            ax.legend(fontsize=7, ncol=2)
    ax = fig.add_subplot(gs[2, :])
    cats = ["Train users", "Test users", "Train files", "Test files"]
    vals = [meta["user"].nunique(), meta_te["user"].nunique(), len(meta), len(meta_te)]
    bars = ax.bar(cats, vals, color=["C0", "C1", "C0", "C1"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, str(v), ha="center")
    ax.set_title("Cross-subject split: 60 labeled train users → 40 unseen test users "
                  "(disjoint user sets)")
    fig.suptitle("Data overview: example sensor traces (one per class) and train/test split",
                  fontsize=12)
    fig.savefig(f"{OUTDIR}/fig02_data_overview.png", dpi=150)
    plt.close(fig)
    print("  fig02_data_overview.png")


# ---------------------------------------------------------------- Section 2
def fig03_loss_curve(X86, y):
    Xi, Xe, yi, ye = train_test_split(X86, y, test_size=INTERNAL_VAL_FRAC,
                                       stratify=y, random_state=L.SEED)
    sw = compute_sample_weight("balanced", yi)
    m = lgb.LGBMClassifier(**L.TUNED)
    m.fit(Xi, yi, sample_weight=sw, eval_set=[(Xi, yi), (Xe, ye)],
          eval_names=["train", "valid"], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    res = m.evals_result_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res["train"]["multi_logloss"], label="train")
    ax.plot(res["valid"]["multi_logloss"], label="validation")
    ax.axvline(m.best_iteration_, color="gray", ls="--",
               label=f"best_iteration = {m.best_iteration_}")
    ax.set_xlabel("boosting iteration"); ax.set_ylabel("multi-logloss")
    ax.set_title("LightGBM training curve (single internal train/val split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig03_loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  fig03_loss_curve.png  (best_iteration={m.best_iteration_})")
    return m


def fig04_feature_importance(model, feature_names):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(20), imp[order][::-1], color="C0")
    ax.set_yticks(range(20))
    ax.set_yticklabels(np.array(feature_names)[order][::-1], fontsize=8)
    ax.set_xlabel("Importance (split count)")
    ax.set_title("Top 20 / 86 feature importances (LightGBM)")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig04_feature_importance.png", dpi=150)
    plt.close(fig)
    print("  fig04_feature_importance.png")


def fig05_hmm(X86, y, fid, user, parts):
    T, prior = L.estimate_transition(y, fid, user)
    s, cv, probs, hmm_macro = parts[0]
    pred_vit = L.smooth(probs, y, fid, user, user, cv, **L.CURRENT)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    im = axes[0].imshow(T, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(6)); axes[0].set_yticks(range(6))
    axes[0].set_xticklabels([f"L{i}" for i in range(6)])
    axes[0].set_yticklabels([f"L{i}" for i in range(6)])
    axes[0].set_xlabel("to state"); axes[0].set_ylabel("from state")
    axes[0].set_title("Estimated transition matrix P(L_t+1 | L_t)")
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                          color="white" if T[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # pick the user with the most distinct labels in their file sequence
    users = np.unique(user)
    best_u, best_var = None, -1
    for u in users:
        v = len(np.unique(y[user == u]))
        if v > best_var:
            best_var, best_u = v, u
    idx = np.where(user == best_u)[0]; idx = idx[np.argsort(fid[idx])]
    raw_pred = probs[idx].argmax(1)
    smooth_pred = pred_vit[idx]
    gt = y[idx]
    t = np.arange(len(idx))
    axes[1].step(t, gt, where="mid", label="ground truth", lw=2.2, color="black")
    axes[1].step(t, raw_pred + 0.12, where="mid", label="raw LGB argmax", lw=1.3, alpha=0.8, color="C1")
    axes[1].step(t, smooth_pred - 0.12, where="mid", label="Viterbi-smoothed", lw=1.3, alpha=0.8, color="C2")
    axes[1].set_yticks(range(6)); axes[1].set_yticklabels([f"L{i}" for i in range(6)])
    axes[1].set_xlabel("file index in user's sequence (time)")
    axes[1].set_ylabel("activity label")
    axes[1].set_title(f"Example file sequence (user {best_u}, {best_var} distinct labels)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig05_hmm_transition_and_sequence.png", dpi=150)
    plt.close(fig)
    print("  fig05_hmm_transition_and_sequence.png")


def aug_member(X_raw, user, k, member, seed_base=L.SEED):
    out = X_raw.copy()
    for u in np.unique(user):
        m = user == u
        seed = (seed_base * 100003 + member * 777767 + k * 9973 + uid(u)) % (2 ** 32)
        out[m] = transform(out[m], np.random.RandomState(seed), ROT, SM, SS)
    return out


def fig06_pseudo_confidence(X_raw, X86, y, user, X86_te):
    """Reproduce pass-1 (5-member, seed_offset=0) of gen_pseudo_aug.py exactly."""
    THRESH = 0.82
    N_PASS1 = 5
    full = dict(TUNED); full["n_estimators"] = 463
    proba = np.zeros((len(X86_te), L.N_CLASSES))
    for e in range(N_PASS1):
        feats, labs = [X86], [y]
        for k in range(K_AUG):
            feats.append(build_features(aug_member(X_raw, user, k, e)))
            labs.append(y)
        Xtr = np.vstack(feats); ytr = np.concatenate(labs)
        p = dict(full); p["random_state"] = L.SEED + e
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        proba += m.predict_proba(X86_te)
        print(f"    pass1 member {e}/{N_PASS1 - 1} done")
    pr1 = proba / N_PASS1
    conf = pr1.max(axis=1)
    mask = conf > THRESH

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(conf[~mask], bins=40, range=(0, 1), color="lightgray", label=f"rejected ({(~mask).sum()})")
    ax.hist(conf[mask], bins=40, range=(0, 1), color="C0", label=f"accepted as pseudo-label ({mask.sum()})")
    ax.axvline(THRESH, color="red", ls="--", label=f"threshold = {THRESH}")
    ax.set_xlabel("max class probability (pass-1 ensemble, 5 members)")
    ax.set_ylabel("# test files")
    ax.set_title(f"Pseudo-label confidence on test set\n"
                  f"{mask.sum()}/{len(conf)} ({mask.mean()*100:.1f}%) accepted, "
                  f"mean conf among accepted = {conf[mask].mean():.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig06_pseudo_confidence.png", dpi=150)
    plt.close(fig)
    print("  fig06_pseudo_confidence.png")


def fig07_per_class_f1_progression(X86, y, fid, user, parts):
    class_mult = np.load("/root/dm-assignment3/cache/class_mult.npy")
    CFG_FB = dict(s=L.CURRENT["s"], e=L.CURRENT["e"], beta=L.CURRENT["beta"],
                   decode="fb", trans=L.CURRENT["trans"], prob_mode=L.CURRENT["prob_mode"])
    raw_pc, vit_pc, cal_pc = [], [], []
    for s, cv, probs, _hmm in parts:
        raw_pred = probs.argmax(1)
        raw_pc.append(f1_score(y, raw_pred, average=None, labels=list(range(6))))
        vit_pred = L.smooth(probs, y, fid, user, user, cv, **L.CURRENT)
        vit_pc.append(f1_score(y, vit_pred, average=None, labels=list(range(6))))
        _, post = L.smooth(probs, y, fid, user, user, cv, return_proba=True, **CFG_FB)
        cal_pred = (post * class_mult).argmax(1)
        cal_pc.append(f1_score(y, cal_pred, average=None, labels=list(range(6))))
    raw_pc = np.array(raw_pc).mean(0)
    vit_pc = np.array(vit_pc).mean(0)
    cal_pc = np.array(cal_pc).mean(0)

    x = np.arange(6); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, raw_pc, w, label=f"raw LGB argmax (mean={raw_pc.mean():.4f})")
    ax.bar(x, vit_pc, w, label=f"+ Viterbi HMM (mean={vit_pc.mean():.4f})")
    ax.bar(x + w, cal_pc, w, label=f"+ HMM(FB) + per-class calib (mean={cal_pc.mean():.4f})")
    ax.set_xticks(x); ax.set_xticklabels([f"L{c}" for c in range(6)])
    ax.set_ylabel("macro-F1 (3-partition mean OOF)")
    ax.set_title("Per-class F1 across decoding stages")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig07_per_class_f1_progression.png", dpi=150)
    plt.close(fig)
    print("  fig07_per_class_f1_progression.png")


# ---------------------------------------------------------------- Section 3
SUBS = {
    "pseudo_aug (best)":   ("submission_pseudo_aug.csv", 0.8061,
                             dict(N_PASS1=5, N_PASS2=7, K_ps=1, thresh=0.82, decode="Viterbi")),
    "pseudo_n9":           ("submission_pseudo_n9.csv", 0.8048,
                             dict(N_PASS1=5, N_PASS2=9, K_ps=1, thresh=0.82, decode="Viterbi")),
    "pseudo (orig)":       ("submission_pseudo.csv", 0.8044,
                             dict(N_PASS1=5, N_PASS2=5, K_ps=0, thresh=0.82, decode="Viterbi")),
    "pseudo_calib":        ("submission_pseudo_calib.csv", 0.8044,
                             dict(N_PASS1=5, N_PASS2=7, K_ps=1, thresh=0.82, decode="FB+class_mult")),
}
BASELINE3 = 0.7088


def fig08_lb_progression():
    names = list(SUBS.keys())
    scores = [SUBS[n][1] for n in names]
    order = np.argsort(scores)[::-1]
    names = [names[i] for i in order]; scores = [scores[i] for i in order]
    all_names = ["Baseline 3"] + names
    all_scores = [BASELINE3] + scores
    colors = ["gray"] + ["C0", "C1", "C2", "C3"][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(all_names, all_scores, color=colors)
    for b, v in zip(bars, all_scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.65, 0.85)
    ax.set_ylabel("Public LB macro-F1")
    ax.set_title("Baseline vs. our submissions")
    ax.axhline(BASELINE3, color="gray", ls="--", lw=1, alpha=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig08_lb_progression.png", dpi=150)
    plt.close(fig)
    print("  fig08_lb_progression.png")


def label_dist(path):
    df = pd.read_csv(path)
    c = df["Label"].value_counts().sort_index()
    return np.array([c.get(i, 0) for i in range(6)])


def fig09_config_and_dist():
    names = list(SUBS.keys())
    dists = {n: label_dist(SUBS[n][0]) for n in names}

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.4)

    # config table
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    cols = ["Submission", "LB", "N_PASS1", "N_PASS2", "K_AUG_PSEUDO", "thresh", "decode"]
    rows = []
    for n in names:
        path, lb, cfg = SUBS[n]
        rows.append([n, f"{lb:.4f}", cfg["N_PASS1"], cfg["N_PASS2"], cfg["K_ps"],
                      cfg["thresh"], cfg["decode"]])
    tbl = ax0.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    ax0.set_title("Top candidates: configuration", fontsize=11, pad=10)

    # grouped bar of label distributions
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(6); w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax1.bar(x + (i - (len(names)-1)/2) * w, dists[n], w, label=f"{n} ({SUBS[n][1]:.4f})")
    ax1.set_xticks(x); ax1.set_xticklabels([f"L{c}" for c in range(6)])
    ax1.set_ylabel("# predicted test files")
    ax1.set_title("Predicted label distribution (test, n=6849)")
    ax1.legend(fontsize=8)
    fig.savefig(f"{OUTDIR}/fig09_config_and_label_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig09_config_and_label_dist.png")


def fig10_agreement_matrix():
    names = list(SUBS.keys())
    preds = {n: pd.read_csv(SUBS[n][0]).sort_values("Id")["Label"].to_numpy() for n in names}
    n = len(names)
    agree = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            agree[i, j] = (preds[a] == preds[b]).mean() * 100

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(agree, cmap="Blues", vmin=95, vmax=100)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agree[i, j]:.2f}%", ha="center", va="center",
                    color="white" if agree[i, j] > 98.5 else "black", fontsize=9)
    ax.set_title("Pairwise prediction agreement (%) on test set")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% identical predictions")
    fig.savefig(f"{OUTDIR}/fig10_agreement_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig10_agreement_matrix.png")


def main():
    print("Loading data...")
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    X86 = build_features(X_raw)
    X_raw_te, meta_te = D.load_split("test")
    X86_te = build_features(X_raw_te)
    feature_names = build_feature_names()

    print("\n=== Section 1: Baseline & problem context ===")
    fig01_class_distribution(meta)
    fig02_data_overview(X_raw, meta, X_raw_te, meta_te)

    print("\n=== Section 2: Experimental pipeline ===")
    model = fig03_loss_curve(X86, y)
    fig04_feature_importance(model, feature_names)
    print("  loading 3 fair partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    fig05_hmm(X86, y, fid, user, parts)
    print("  fig06: training pass-1 ensemble (5 members)...")
    fig06_pseudo_confidence(X_raw, X86, y, user, X86_te)
    fig07_per_class_f1_progression(X86, y, fid, user, parts)

    print("\n=== Section 3: Baseline vs top-3 ===")
    fig08_lb_progression()
    fig09_config_and_dist()
    fig10_agreement_matrix()

    print(f"\nAll figures written to {OUTDIR}/")


if __name__ == "__main__":
    main()
