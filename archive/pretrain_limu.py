"""Idea C -- LIMU-BERT-style self-supervised pretraining (proper version).

Improves on the crude Idea-1 autoencoder with the LIMU-BERT recipe (Xu et al.):
  - feature-dimension expansion (6 -> d_model) + per-channel normalization
  - Transformer encoder over the 300 timesteps
  - BERT-style probabilistic masking (15% of steps; of those 80% zeroed, 10% random,
    10% kept) and reconstruct the masked readings
Pretrains on ALL 17,869 files (train+test, no labels). Saves embeddings to the SAME files
stepSSL.py reads, so run:  python pretrain_limu.py  &&  python stepSSL.py

GPU; seeded. Embeddings saved -> downstream gate reproducible.
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
D_MODEL, NHEAD, LAYERS, FF = 72, 4, 3, 144
BATCH, EPOCHS, LR = 256, 60, 1e-3
MASK_P = 0.15


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class LIMUBert(nn.Module):
    def __init__(self, ch=6, d=D_MODEL, maxlen=300):
        super().__init__()
        self.proj = nn.Linear(ch, d)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, d))
        enc = nn.TransformerEncoderLayer(d, NHEAD, FF, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, LAYERS)
        self.head = nn.Linear(d, ch)          # reconstruct readings

    def encode(self, x):                       # x:(B,T,6)
        h = self.enc(self.proj(x) + self.pos[:, :x.size(1)])
        return torch.cat([h.mean(1), h.amax(1)], dim=1)   # (B,2d) file embedding

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos[:, :x.size(1)])
        return self.head(h)                    # (B,T,6)


def bert_mask(x, gen):
    """BERT-style mask. Returns (corrupted_input, mask_bool[B,T])."""
    B, T, C = x.shape
    m = torch.rand(B, T, generator=gen) < MASK_P
    xin = x.clone()
    r = torch.rand(B, T, generator=gen)
    zero = m & (r < 0.8)
    rand = m & (r >= 0.8) & (r < 0.9)         # 10% random
    xin[zero] = 0.0
    if rand.any():
        xin[rand] = torch.randn(int(rand.sum()), C, generator=gen)
    return xin, m


def main():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(CACHE, exist_ok=True)

    Xtr, _ = D.load_split("train"); Xte, _ = D.load_split("test")
    allX = np.concatenate([Xtr, Xte], axis=0)
    mu = allX.reshape(-1, 6).mean(0); sd = allX.reshape(-1, 6).std(0) + 1e-8
    data = torch.tensor((allX - mu) / sd, dtype=torch.float32)   # (N,300,6)
    print(f"LIMU-BERT pretrain on {len(data)} files, device={DEVICE}, d_model={D_MODEL}")

    model = LIMUBert().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(data), generator=gen).numpy()
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i + BATCH]
            xb = data[bi]
            xin, m = bert_mask(xb, gen)
            xb, xin, m = xb.to(DEVICE), xin.to(DEVICE), m.to(DEVICE)
            rec = model(xin)
            loss = ((rec - xb)[m] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}: masked MSE = {tot/len(data):.4f}")

    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(data), 512):
            embs.append(model.encode(data[i:i + 512].to(DEVICE)).cpu().numpy())
    emb = np.concatenate(embs, 0)
    np.savez_compressed(os.path.join(CACHE, "ssl_emb_train.npz"), e=emb[:len(Xtr)])
    np.savez_compressed(os.path.join(CACHE, "ssl_emb_test.npz"), e=emb[len(Xtr):])
    print(f"saved LIMU-BERT embeddings: train {emb[:len(Xtr)].shape}, test {emb[len(Xtr):].shape} "
          f"-> {CACHE}/ssl_emb_*.npz  (now run: python stepSSL.py)")


if __name__ == "__main__":
    main()
