"""Shared library for decoder-on-frozen-LGB-emissions experiments (A/B/C/D).

Frozen: LightGBM OOF class probabilities (6-dim emissions). We only swap the DECODER
(HMM vs BiGRU+CRF vs Mamba vs Transformer) over each user's file sequence.

Partitioning: StratifiedGroupKFold collapses to ~2 distinct splits regardless of seed,
so we use a custom random group-partitioner (shuffle 60 users by seed -> 5 folds of 12)
to get 3 TRULY-distinct partitions [0,1,2]. The HMM reference is recomputed on these.

Determinism: CPU, full-batch padded training, all seeds pinned -> bit-identical reruns.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import temporal_lib as L
from har_cv import SEED, N_CLASSES

DEVICE = "cpu"
NOISE = 0.0024
PART_SEEDS = [0, 1, 2]
CACHE = "/root/dm-assignment3/cache"


class GroupPartition:
    """Random group K-fold: shuffle users by seed, split into n_splits folds."""
    def __init__(self, seed, n_splits=5):
        self.seed, self.n = seed, n_splits

    def split(self, X, y, groups):
        users = np.unique(groups)
        perm = np.random.RandomState(self.seed).permutation(users)
        for f in np.array_split(perm, self.n):
            vm = np.isin(groups, f)
            yield np.where(~vm)[0], np.where(vm)[0]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_partitions(X, y, fid, user):
    """Return list of (seed, cv, lgb_probs, hmm_macro) for the 3 distinct partitions."""
    os.makedirs(CACHE, exist_ok=True)
    groups = user
    out = []
    for s in PART_SEEDS:
        cv = GroupPartition(s)
        path = os.path.join(CACHE, f"lgbprobs_gp{s}.npz")
        if os.path.exists(path):
            probs = np.load(path)["p"]
        else:
            probs = L.lgb_oof_probs(X, y, groups, cv)
            np.savez_compressed(path, p=probs)
        hmm = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        out.append((s, cv, probs, f1_score(y, hmm, average="macro")))
    return out


# ---------- inputs ----------
def input6(probs):
    return probs


def input12(probs):
    oh = np.zeros_like(probs)
    oh[np.arange(len(probs)), probs.argmax(1)] = 1.0
    return np.hstack([probs, oh])


# ---------- sequence batching ----------
def user_sequences(fid, user):
    return {u: np.where(user == u)[0][np.argsort(fid[user == u])] for u in np.unique(user)}


def build_batch(users, seqs, Xin, y):
    rows = [seqs[u] for u in users]
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    Lmax, D = int(lengths.max()), Xin.shape[1]
    Xpad = torch.zeros(len(users), Lmax, D)
    Ypad = torch.zeros(len(users), Lmax, dtype=torch.long)   # 0-pad (masked out)
    mask = torch.zeros(len(users), Lmax, dtype=torch.bool)
    for i, r in enumerate(rows):
        l = len(r)
        Xpad[i, :l] = torch.tensor(Xin[r], dtype=torch.float32)
        Ypad[i, :l] = torch.tensor(y[r], dtype=torch.long)
        mask[i, :l] = True
    return Xpad, Ypad, lengths, mask, rows


# ---------- generic train/predict for one fold ----------
def train_fold(model_fn, is_crf, Xin, y, seqs, tr_users, va_users,
               max_epochs=200, patience=20, lr=1e-3, wd=1e-3, seed=SEED):
    set_seed(seed)
    train_rows = np.concatenate([seqs[u] for u in tr_users])
    Xtr, Ytr, Ltr, Mtr, _ = build_batch(tr_users, seqs, Xin, y)
    Xva, Yva, Lva, Mva, va_rows = build_batch(va_users, seqs, Xin, y)
    yva = y[np.concatenate(va_rows)]
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(N_CLASSES),
                                           y=y[train_rows]), dtype=torch.float32)
    model = model_fn(Xin.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_f1, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        model.train(); opt.zero_grad()
        if is_crf:
            em = model.emissions(Xtr, Ltr)
            # token_mean keeps the CRF NLL per-token, comparable to the weighted CE so the
            # inverse-freq class weighting actually counters majority-class collapse.
            loss = -model.crf(em, Ytr, mask=Mtr, reduction="token_mean") \
                + F.cross_entropy(em[Mtr], Ytr[Mtr], weight=cw)
        else:
            logits = model(Xtr, Ltr)
            loss = F.cross_entropy(logits[Mtr], Ytr[Mtr], weight=cw)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            if is_crf:
                em = model.emissions(Xva, Lva)
                paths = model.crf.decode(em, mask=Mva)
                pv = np.concatenate([np.array(p) for p in paths])
            else:
                pv = model(Xva, Lva)[Mva].argmax(1).numpy()
        f = f1_score(yva, pv, average="macro")
        if f > best_f1 + 1e-5:
            best_f1, since = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        if is_crf:
            em = model.emissions(Xva, Lva)
            paths = model.crf.decode(em, mask=Mva)
            preds = np.concatenate([np.array(p) for p in paths])
            probs = torch.softmax(em, dim=2)[Mva].numpy()
        else:
            logits = model(Xva, Lva)
            probs = torch.softmax(logits, dim=2)[Mva].numpy()
            preds = probs.argmax(1)
    return np.concatenate(va_rows), preds, probs


def run_decoder(model_fn, is_crf, input_fn, parts, X, y, fid, user, **train_kw):
    """Run a decoder across the 3 partitions. Returns (macros, oof_pred0, oof_proba0)."""
    groups = user
    seqs = user_sequences(fid, user)
    macros, oof0_pred, oof0_proba = [], None, None
    for i, (s, cv, probs, _hmm) in enumerate(parts):
        Xin = input_fn(probs)
        oof_pred = np.full(len(y), -1, dtype=int)
        oof_proba = np.zeros((len(y), N_CLASSES))
        for tr, va in cv.split(X, y, groups):
            tru = sorted(set(user[tr])); vau = sorted(set(user[va]))
            idx, pr, pb = train_fold(model_fn, is_crf, Xin, y, seqs, tru, vau, **train_kw)
            oof_pred[idx] = pr; oof_proba[idx] = pb
        macros.append(f1_score(y, oof_pred, average="macro"))
        if i == 0:
            oof0_pred, oof0_proba = oof_pred, oof_proba
    return macros, oof0_pred, oof0_proba


def gate(dec_macros, hmm_macros):
    dm, hm = np.array(dec_macros), np.array(hmm_macros)
    return bool(dm.mean() - hm.mean() > NOISE and np.all(dm >= hm - 1e-9))


def diagnostics(y, pred, tag):
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    p2 = precision_score(y, pred, labels=[2], average="micro", zero_division=0)
    r2 = recall_score(y, pred, labels=[2], average="micro", zero_division=0)
    print(f"  [{tag}] true1->pred2={cm[1,2]} (HMM had 78) | L2 prec={p2:.4f} rec={r2:.4f}")
    return cm


# ---------- models ----------
class BiGRUCRF(nn.Module):
    def __init__(self, in_dim, hidden=32, drop=0.3):
        super().__init__()
        from torchcrf import CRF
        self.gru = nn.GRU(in_dim, hidden, 1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.lin = nn.Linear(2 * hidden, N_CLASSES)
        self.crf = CRF(N_CLASSES, batch_first=True)

    def emissions(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.lin(self.drop(h))


class MambaBlock(nn.Module):
    """Compact pure-PyTorch selective SSM (Mamba, Gu & Dao 2024), CPU-deterministic.

    The official mamba-ssm requires custom CUDA kernels (GPU-only, non-deterministic),
    which conflicts with our reproducibility gate -- this is a faithful minimal S6 block.
    Unidirectional (causal); padding sits at sequence end so it cannot leak backward.
    """
    def __init__(self, d_model=32, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.d_state = d_state

    def forward(self, x):                       # x: (B, Lseq, d_model)
        B_, Lq, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)             # (B,L,d_inner) each
        xi = self.conv1d(xi.transpose(1, 2))[:, :, :Lq].transpose(1, 2)
        xi = F.silu(xi)
        dbl = self.x_proj(xi)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))       # (B,L,d_inner)
        A = -torch.exp(self.A_log)              # (d_inner,d_state)
        dA = torch.exp(dt[..., None] * A[None, None])          # (B,L,d_inner,N)
        dBu = dt[..., None] * Bm[:, :, None, :] * xi[..., None]  # (B,L,d_inner,N)
        h = torch.zeros(B_, self.d_inner, self.d_state)
        ys = []
        for t in range(Lq):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t, None, :]).sum(-1))         # (B,d_inner)
        y = torch.stack(ys, dim=1) + xi * self.D
        return self.out_proj(y * F.silu(z))


class MambaSeq(nn.Module):
    def __init__(self, in_dim, d_model=32, d_state=16):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_model, d_state)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x, lengths):
        h = self.proj(x)
        h = h + self.block(self.norm(h))
        return self.head(h)


class TransformerSeq(nn.Module):
    def __init__(self, in_dim, d_model=32, nhead=2, layers=2, ff=32, drop=0.2, maxlen=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x, lengths):
        Lmax = x.size(1)
        h = self.proj(x) + self.pos[:, :Lmax]
        key_pad = torch.arange(Lmax)[None, :] >= lengths[:, None]
        h = self.enc(h, src_key_padding_mask=key_pad)
        return self.head(h)
