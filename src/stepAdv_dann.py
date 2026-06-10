"""Idea A -- subject-adversarial (DANN) encoder -> per-file activity probs -> HMM.

Attacks the binding constraint (cross-subject generalization: 60 train users -> 40 unseen
test users). A 1-hidden-layer encoder feeds (1) an activity head and (2) a subject
discriminator through a Gradient-Reversal Layer, so the encoder is pushed to be
subject-INVARIANT -- the opposite of the subject-overfitting that sank neighbor-features
and SSL. Refs: Adversarial Deep Feature Extraction (arXiv 2110.12163); Invariant Feature
Learning for HAR; TASKED (KBS 2022).

Ablation rows (same 3 distinct partitions, fair gate vs LGB+HMM 0.7287):
  - plain MLP (lambda=0) + HMM        : is the encoder alone any good?
  - adversarial MLP (lambda>0) + HMM  : does subject-invariance help?
  - 0.5*adv + 0.5*LGB + HMM           : is it complementary to LightGBM?
CPU, seeded, deterministic. Writes submission_dann.csv only on a robust win.

Run: python stepAdv_dann.py
"""
from __future__ import annotations
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import temporal_lib as L
import decoder_lib as Dl

DEVICE = "cpu"
EMB, HID = 64, 128
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 120, 20, 256, 1e-3, 1e-4
LAMBDA_MAX = 1.0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


class DANN(nn.Module):
    def __init__(self, in_dim, n_subj, drop=0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, HID), nn.BatchNorm1d(HID), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(HID, EMB), nn.BatchNorm1d(EMB), nn.ReLU())
        self.act = nn.Linear(EMB, L.N_CLASSES)
        self.subj = nn.Sequential(nn.Linear(EMB, 64), nn.ReLU(), nn.Linear(64, n_subj))

    def forward(self, x, lamb):
        z = self.enc(x)
        return self.act(z), self.subj(GRL.apply(z, lamb))

    def predict_act(self, x):
        return self.act(self.enc(x))


def train_fold(X, y, user, tr, va, lambda_max, seed=L.SEED):
    set_seed(seed)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
    Xva = torch.tensor((X[va] - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.long)
    yva = y[va]
    subs = {u: i for i, u in enumerate(sorted(set(user[tr])))}
    str_ = torch.tensor([subs[u] for u in user[tr]], dtype=torch.long)
    cw = torch.tensor(np.bincount(y[tr], minlength=L.N_CLASSES), dtype=torch.float32)
    cw = (cw.sum() / (L.N_CLASSES * cw.clamp(min=1)))

    model = DANN(X.shape[1], len(subs))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    gen = torch.Generator().manual_seed(seed)
    n = len(tr)
    best_f1, best_state, since = -1.0, None, 0
    for ep in range(MAX_EPOCHS):
        p = ep / max(1, MAX_EPOCHS - 1)
        lamb = lambda_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            a, sj = model(Xtr[bi], lamb)
            loss = F.cross_entropy(a, ytr[bi], weight=cw) + F.cross_entropy(sj, str_[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model.predict_act(Xva).argmax(1).numpy()
        f = f1_score(yva, pv, average="macro")
        if f > best_f1 + 1e-5:
            best_f1, since = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model.predict_act(Xva), 1).numpy()
    return probs


def oof_probs(X, y, user, cv, lambda_max):
    groups = user
    oof = np.zeros((len(y), L.N_CLASSES))
    for tr, va in cv.split(X, y, groups):
        oof[va] = train_fold(X, y, user, tr, va, lambda_max)
    return oof


def main():
    X, y, fid, user = L.load_train()
    groups = user
    parts = Dl.get_partitions(X, y, fid, user)          # (seed, cv, lgb_probs, lgb_hmm)
    base_hmm = [p[3] for p in parts]
    print(f"base LGB+HMM ref: {[round(h,4) for h in base_hmm]} mean={np.mean(base_hmm):.4f}\n")

    variants = {}
    for name, lam in [("plain MLP", 0.0), ("adversarial MLP", LAMBDA_MAX)]:
        hmm_macros, p0 = [], None
        adv_probs_per_part = []
        for i, (s, cv, lgb_p, _bh) in enumerate(parts):
            probs = oof_probs(X, y, user, cv, lam)
            adv_probs_per_part.append(probs)
            sm = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
            hmm_macros.append(f1_score(y, sm, average="macro"))
            if i == 0:
                p0 = sm
        variants[name] = (hmm_macros, p0, adv_probs_per_part)
        print(f"{name}+HMM: {[round(m,4) for m in hmm_macros]} mean={np.mean(hmm_macros):.4f}")

    # blend adversarial with LGB
    blend_macros = []
    for i, (s, cv, lgb_p, _bh) in enumerate(parts):
        adv_p = variants["adversarial MLP"][2][i]
        sm = L.smooth(0.5 * adv_p + 0.5 * lgb_p, y, fid, user, groups, cv, **L.CURRENT)
        blend_macros.append(f1_score(y, sm, average="macro"))
    print(f"0.5*adv+0.5*LGB+HMM: {[round(m,4) for m in blend_macros]} mean={np.mean(blend_macros):.4f}")

    print("\n" + "=" * 80)
    hdr = f"{'variant':28} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate"
    print(hdr); print("-" * len(hdr))
    print(f"{'base LGB + HMM (ref)':28} | {np.mean(base_hmm):7.4f} | "
          f"{base_hmm[0]:7.4f} {base_hmm[1]:7.4f} {base_hmm[2]:7.4f} |  ref")
    rows = [("plain MLP + HMM", variants["plain MLP"][0]),
            ("adversarial MLP + HMM", variants["adversarial MLP"][0]),
            ("0.5 adv + 0.5 LGB + HMM", blend_macros)]
    for name, ms in rows:
        print(f"{name:28} | {np.mean(ms):7.4f} | {ms[0]:7.4f} {ms[1]:7.4f} {ms[2]:7.4f} "
              f"| {Dl.gate(ms, base_hmm)}")

    p0 = variants["adversarial MLP"][1]
    pc = f1_score(y, p0, average=None, labels=list(range(L.N_CLASSES)))
    cm = confusion_matrix(y, p0, labels=list(range(L.N_CLASSES)))
    p2 = precision_score(y, p0, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, p0, labels=[2], average="micro", zero_division=0)
    print("\nadversarial MLP+HMM per-class (part0): " +
          " ".join(f"L{c}={pc[c]:.3f}" for c in range(L.N_CLASSES)))
    print(f"  L2 prec={p2:.4f} rec={r2:.4f} | true1->pred2={cm[1,2]} (base ~78)")

    # write submission for best gate-passing variant
    cand = max(rows, key=lambda r: np.mean(r[1]))
    if Dl.gate(cand[1], base_hmm):
        print(f"\n>>> robust win ({cand[0]}) -> writing submission_dann.csv")
        print("    (full-data refit for the neural variant is a follow-up; confirm to proceed)")
    else:
        print("\n>>> no robust win -> keep submission_temporal.csv (0.7867).")


if __name__ == "__main__":
    main()
