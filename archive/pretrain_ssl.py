"""Idea 1, step 1 -- transductive self-supervised pretraining.

Masked-reconstruction autoencoder over the raw 300x6 sequences of ALL 17,869 files
(train + test, NO labels). The encoder learns the motion manifold of the TEST users too
(only their features, never labels -> no leakage), directly shrinking the train->test
generalization gap that caps our OOF. Saves per-file embeddings for the downstream
LightGBM+HMM (deterministic once embeddings are cached).

GPU (RTX 3090); seeded. Embeddings are saved, so the downstream gate is reproducible.
Run: python pretrain_ssl.py
"""
from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn

import har_data as D
from har_cv import SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "/root/dm-assignment3/cache"
EMB_DIM = 64
BATCH, EPOCHS, LR = 256, 60, 1e-3
MASK_RATIO, SPAN = 0.30, 15


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ConvAE(nn.Module):
    def __init__(self, ch=6, h=EMB_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(ch, h, 7, padding=3), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Conv1d(h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(h, h, 5, padding=2), nn.ReLU(),
            nn.Conv1d(h, ch, 7, padding=3),
        )

    def encode(self, x):                  # x: (B,6,T)
        z = self.enc(x)                   # (B,h,T)
        return torch.cat([z.mean(-1), z.amax(-1)], dim=1)   # (B,2h) file embedding

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)                # (B,6,T)


def make_mask(n, T, gen):
    """Span mask: True where corrupted (to reconstruct)."""
    m = torch.zeros(n, T, dtype=torch.bool)
    n_spans = max(1, int(MASK_RATIO * T / SPAN))
    starts = torch.randint(0, T - SPAN, (n, n_spans), generator=gen)
    for i in range(n):
        for s in starts[i]:
            m[i, s:s + SPAN] = True
    return m


def main():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(CACHE, exist_ok=True)

    Xtr, _ = D.load_split("train")        # (n,300,6)
    Xte, _ = D.load_split("test")
    allX = np.concatenate([Xtr, Xte], axis=0)
    mu = allX.reshape(-1, 6).mean(0); sd = allX.reshape(-1, 6).std(0) + 1e-8
    alln = ((allX - mu) / sd).transpose(0, 2, 1)          # (N,6,300)
    data = torch.tensor(alln, dtype=torch.float32)
    print(f"pretraining on {len(data)} files (train {len(Xtr)} + test {len(Xte)}), device={DEVICE}")

    model = ConvAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(SEED)
    idx_all = np.arange(len(data))

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(data), generator=gen).numpy()
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i + BATCH]
            xb = data[bi].to(DEVICE)
            m = make_mask(len(bi), xb.size(2), gen).to(DEVICE)   # (B,T)
            xin = xb.clone(); xin[:, :, :] = xb
            xin = xin.masked_fill(m.unsqueeze(1), 0.0)
            rec = model(xin)
            mm = m.unsqueeze(1).expand_as(xb)
            loss = ((rec - xb)[mm] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}: masked-recon MSE = {tot/len(data):.4f}")

    # extract embeddings (deterministic eval pass, no masking)
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(data), 512):
            embs.append(model.encode(data[i:i + 512].to(DEVICE)).cpu().numpy())
    emb = np.concatenate(embs, axis=0)
    np.savez_compressed(os.path.join(CACHE, "ssl_emb_train.npz"), e=emb[:len(Xtr)])
    np.savez_compressed(os.path.join(CACHE, "ssl_emb_test.npz"), e=emb[len(Xtr):])
    print(f"saved embeddings: train {emb[:len(Xtr)].shape}, test {emb[len(Xtr):].shape} "
          f"(dim={emb.shape[1]}) -> {CACHE}/ssl_emb_*.npz")


if __name__ == "__main__":
    main()
