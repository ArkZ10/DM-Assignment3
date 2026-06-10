"""1D ResNet on raw 300x6 sequences with cross-subject augmentation -- new information axis.

Per-file statistics (86 feats) capture WHAT a signal looks like on average. They discard
temporal ORDER -- how channels rise/fall and co-vary across the 300 timesteps within a
file (e.g., step-rhythm periodicity, acceleration trends, cross-axis covariance dynamics).
A 1D ResNet on the raw sequence captures all of that directly, without hand-crafting it.

Combined with the same proven augmentation recipe (3D gravity rotation + magnitude
scaling per synthetic subject), this gives a model that is both:
  - architecturally different from LGB (gradient-flow + convolution vs. tree splits)
  - informationally different (raw temporal order vs. per-file aggregates)

These two properties make it a genuinely complementary ensemble member -- not more
correlated noise on the same features.

Fair 3-partition gate vs frozen 86-feat LGB+HMM (0.7287 ref). GPU (RTX 3090).
Run: python stepResNet.py
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import har_data as D
import temporal_lib as L
import decoder_lib as Dl
from stepAugTTA import transform, uid, K_AUG, ROT, SM, SS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = L.N_CLASSES
BATCH = 256
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WD = 1e-4
CACHE_PROBS = "/root/dm-assignment3/cache/resnet_oof_probs.npz"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---- augmentation (same recipe as the proven LGB aug) ----
def aug_batch(Xr, rng, deg, sm, ss):
    """Xr: (B,300,6) numpy -> augmented (B,300,6) numpy."""
    mean = (Xr[..., :3] @ __import__('stepAugTTA').rot_mat(rng, deg).T) * rng.uniform(1-sm, 1+sm)
    std = np.abs(Xr[..., 3:] * rng.uniform(1-ss, 1+ss))
    return np.concatenate([mean, std], axis=-1) + rng.normal(0, 0.01, Xr.shape)


def make_aug_dataset(X_raw, y, user, k_aug=K_AUG):
    """Concatenate real + K augmented copies; per-user deterministic seeding."""
    Xs, ys = [X_raw], [y]
    for k in range(k_aug):
        Xk = X_raw.copy()
        for u in np.unique(user):
            m = user == u
            seed = (L.SEED * 100003 + k * 9973 + uid(u)) % (2**32)
            rng = np.random.RandomState(seed)
            Xk[m] = aug_batch(X_raw[m], rng, ROT, SM, SS)
        Xs.append(Xk); ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


# ---- model ----
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, in_ch=6, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock1D(in_ch, 64, 8),
            ResBlock1D(64, 128, 5),
            ResBlock1D(128, 128, 3),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):              # x: (B, 6, T)
        return self.head(self.net(x).mean(dim=2))


# ---- per-fold training ----
def train_fold(X_raw, y, user, tr, va, seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # per-fold standardisation using train-fold stats (no leakage)
    mu = X_raw[tr].reshape(-1, 6).mean(0); sd = X_raw[tr].reshape(-1, 6).std(0) + 1e-8
    Xn = (X_raw - mu) / sd                 # normalise full array; model only sees fold-correct data

    # augmented training set
    Xaug, yaug = make_aug_dataset(Xn[tr], y[tr], user[tr])
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=yaug)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    gen = torch.Generator(); gen.manual_seed(seed)
    Xt = torch.tensor(Xaug.transpose(0, 2, 1), dtype=torch.float32)
    yt = torch.tensor(yaug, dtype=torch.long)
    Xv = torch.tensor(Xn[va].transpose(0, 2, 1), dtype=torch.float32).to(DEVICE)

    model = ResNet1D().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_f1, best_state, since = -1.0, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i + BATCH]
            xb, yb = Xt[bi].to(DEVICE), yt[bi].to(DEVICE)
            loss = F.cross_entropy(model(xb), yb, weight=cw_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).cpu().numpy()
        f = f1_score(y[va], pv, average="macro")
        if f > best_f1 + 1e-5:
            best_f1, since = f, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xv), 1).cpu().numpy()
    return probs, best_f1


def oof_probs(X_raw, y, user, cv, verbose=True):
    oof = np.zeros((len(y), N_CLASSES))
    for fold_i, (tr, va) in enumerate(cv.split(X_raw, y, user)):
        probs, bf1 = train_fold(X_raw, y, user, tr, va, L.SEED + fold_i)
        oof[va] = probs
        if verbose:
            print(f"    fold {fold_i}: best val macro-F1 = {bf1:.4f}")
    return oof


def main():
    X_raw, meta = D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    from step2_familyB import build_features
    X86 = build_features(X_raw)

    print(f"Device: {DEVICE}  |  shapes: X_raw={X_raw.shape}, y={y.shape}\n")

    print("Reference: frozen 86-feat LGB+HMM on 3 distinct partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)
    base_ms = np.array([p[3] for p in parts])
    print(f"  86-feat mean = {base_ms.mean():.4f}  {[round(m, 4) for m in base_ms]}\n")

    new_ms = []; all_probs = []
    for s, cv, _lgb_p, _bh in parts:
        print(f"partition {s}: training ResNet (augmented, {MAX_EPOCHS} epochs max, patience {PATIENCE})...")
        probs = oof_probs(X_raw, y, user, cv)
        all_probs.append(probs)
        pred = L.smooth(probs, y, fid, user, user, cv, **L.CURRENT)
        m = f1_score(y, pred, average="macro")
        new_ms.append(m)
        print(f"  -> partition {s} macro-F1 = {m:.4f}\n")
    new_ms = np.array(new_ms)

    print("=" * 60)
    print(f"ResNet+HMM mean = {new_ms.mean():.4f}  {[round(m, 4) for m in new_ms]}")
    print(f"delta mean: {new_ms.mean() - base_ms.mean():+.4f}   (noise floor {Dl.NOISE})")
    win = Dl.gate(new_ms, base_ms)
    print(f"ROBUST WIN vs 86-feat LGB+HMM: {win}")

    # always save partition-0 OOF for downstream blending experiments
    np.savez_compressed(CACHE_PROBS, probs=all_probs[0], y=y, fid=fid, user=user)
    print(f"\nSaved partition-0 OOF probs -> {CACHE_PROBS}")
    print("Next: run gen_resnet_lgb_ens.py to blend ResNet + LGB aug ensemble.")


if __name__ == "__main__":
    main()
