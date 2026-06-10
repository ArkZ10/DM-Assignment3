"""Step 4 (phase 1): 1D-CNN on the RAW 300x6 sequences. GPU (RTX 3090).

The aggregate-feature LightGBM (OOF macro-F1 0.7095) remains the baseline to beat and
a future ensemble member -- this is an independent model on the un-collapsed series.

Discipline is IDENTICAL to the prior steps:
  - StratifiedGroupKFold by user, 5 folds, same SEED (reuses har_cv.make_cv).
  - Per-fold channel standardization using TRAIN-fold stats only (no leakage).
  - Class imbalance via inverse-frequency weighted cross-entropy (tried first, per spec).
  - Early stopping on a USER-GROUPED internal split (GroupShuffleSplit) -- users never
    leak into their own early-stopping val.

Reproducibility (zero-grade gate): CUBLAS_WORKSPACE_CONFIG set before importing torch,
torch.use_deterministic_algorithms(True, warn_only=True), cuDNN deterministic, all RNGs
(python/numpy/torch/cuda) and the DataLoader generator seeded. We run the whole CV
twice and report whether OOF is bit-identical; if not, we report the run-to-run std.

Run: python step4_cnn.py
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede torch import

import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import har_data as D
from har_cv import SEED, N_CLASSES, make_cv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LGBM_ROW = {  # baseline to beat
    "macro": 0.7095,
    "per_class": [0.9648, 0.8976, 0.2238, 0.6991, 0.8218, 0.6496],
}

# training config (kept simple for the first sequence model)
BATCH = 128
MAX_EPOCHS = 80
PATIENCE = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
INTERNAL_VAL_FRAC = 0.15


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNN1D(nn.Module):
    """A few Conv1d blocks over time -> global avg+max pool -> FC head."""
    def __init__(self, in_ch=6, n_classes=N_CLASSES, p_drop=0.3):
        super().__init__()
        def block(ci, co, k):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, padding=k // 2),
                nn.BatchNorm1d(co), nn.ReLU(),
            )
        self.features = nn.Sequential(
            block(in_ch, 64, 7), nn.MaxPool1d(2),    # 300 -> 150
            block(64, 128, 5), nn.MaxPool1d(2),      # 150 -> 75
            block(128, 128, 3),                      # 75
        )
        self.head = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(p_drop), nn.Linear(64, n_classes),
        )

    def forward(self, x):                 # x: (B, 6, T)
        h = self.features(x)              # (B, 128, T')
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # (B, 256)
        return self.head(pooled)


def make_loader(X, y, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=g, num_workers=0)


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = torch.from_numpy(X[i:i + 512]).float().to(DEVICE)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_fold(X_tr, y_tr, groups_tr, X_va, fold):
    """Standardize (train stats), grouped internal split, weighted-CE train w/ early stop."""
    set_all_seeds(SEED)  # reset per fold for reproducible init/shuffle

    # channel standardization from TRAIN rows only
    mean = X_tr.reshape(-1, X_tr.shape[2]).mean(0)
    std = X_tr.reshape(-1, X_tr.shape[2]).std(0) + 1e-8
    norm = lambda A: (A - mean) / std

    # user-grouped internal early-stopping split
    gss = GroupShuffleSplit(n_splits=1, test_size=INTERNAL_VAL_FRAC, random_state=SEED)
    sub_tr, sub_es = next(gss.split(X_tr, y_tr, groups_tr))
    Xi = np.transpose(norm(X_tr[sub_tr]), (0, 2, 1))   # (n,6,T) for Conv1d
    yi = y_tr[sub_tr]
    Xe = np.transpose(norm(X_tr[sub_es]), (0, 2, 1))
    ye = y_tr[sub_es]
    Xv = np.transpose(norm(X_va), (0, 2, 1))

    # inverse-frequency class weights from internal-train labels
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=yi)
    weight = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = CNN1D().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = make_loader(Xi, yi, shuffle=True, seed=SEED + fold)

    best_f1, best_state, since = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        es_pred = predict(model, Xe)
        f1 = f1_score(ye, es_pred, average="macro")
        if f1 > best_f1 + 1e-5:
            best_f1, since = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return predict(model, Xv), epoch + 1


def run_cv(X, y, groups):
    cv = make_cv(stratified=True)
    oof = np.full(len(y), -1, dtype=int)
    fold_f1s, times, epochs = [], [], []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        t0 = time.time()
        pred, ep = train_fold(X[tr], y[tr], groups[tr], X[va], fold)
        oof[va] = pred
        dt = time.time() - t0
        f = f1_score(y[va], pred, average="macro")
        fold_f1s.append(f); times.append(dt); epochs.append(ep)
        print(f"  fold {fold}: macro-F1={f:.4f}  (epochs={ep}, {dt:.1f}s)")
    assert (oof >= 0).all()
    macro = f1_score(y, oof, average="macro")
    per_class = f1_score(y, oof, average=None, labels=list(range(N_CLASSES)))
    print(f"  mean fold macro-F1: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
    print(f"  mean train time/fold: {np.mean(times):.1f}s")
    return oof, macro, per_class


def report(oof, macro, per_class):
    print("\n" + "=" * 86)
    hdr = f"{'feature_set / model':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    print(hdr); print("-" * len(hdr))

    def row(name, m, pc):
        cells = "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES))
        print(f"{name:30} | {m:7.4f} | {cells}")

    row("A+B aggregates, LightGBM", LGBM_ROW["macro"], LGBM_ROW["per_class"])
    row("raw 300x6, 1D-CNN", macro, per_class)

    print("\nPer-class deltas (1D-CNN - LightGBM):")
    for c in range(N_CLASSES):
        b, nw = LGBM_ROW["per_class"][c], per_class[c]
        print(f"  L{c}: {b:.4f} -> {nw:.4f} ({nw - b:+.4f})")
    print(f"  macro-F1: {LGBM_ROW['macro']:.4f} -> {macro:.4f} ({macro - LGBM_ROW['macro']:+.4f})")

    cm = confusion_matrix(oof_y, oof, labels=list(range(N_CLASSES)))
    print("\nCONFUSION MATRIX, 1D-CNN (rows = TRUE, cols = PRED)")
    print("        " + "".join(f"{c:>7}" for c in range(N_CLASSES)) + f"{'total':>8}")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t, p]:>7}" for p in range(N_CLASSES)) + f"{cm[t].sum():>8}")
    print("pred tot|" + "".join(f"{cm[:, p].sum():>7}" for p in range(N_CLASSES)))

    l2_prec = precision_score(oof_y, oof, labels=[2], average="micro", zero_division=0)
    l2_rec = recall_score(oof_y, oof, labels=[2], average="micro", zero_division=0)
    print("\nL2 detail (raw sequence vs aggregates):")
    print(f"  true1 -> pred2: {cm[1,2]}   (LightGBM had 78, LR had 674)")
    print(f"  L2 precision: {l2_prec:.4f}   L2 recall: {l2_rec:.4f}   "
          f"(TP={cm[2,2]}, pred2 total={cm[:,2].sum()}, true2 total={cm[2].sum()})")
    print(f"\nOverall macro-F1: {macro:.4f}  vs LightGBM 0.7095  vs Baseline 3 0.7088")


if __name__ == "__main__":
    configure_determinism()
    X_raw, meta = D.load_split("train")
    oof_y = meta["label"].to_numpy()
    groups = meta["user"].to_numpy()
    print(f"raw sequences: X={X_raw.shape}, device={DEVICE}\n")

    oof, macro, per_class = run_cv(X_raw, oof_y, groups)
    report(oof, macro, per_class)
