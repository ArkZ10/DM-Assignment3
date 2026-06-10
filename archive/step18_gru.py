"""Cross-file autoregressive GRU: sequence-to-sequence classification at the FILE level.

Each user = one sequence of files (sorted by file_id = recording order). A BiGRU reads
the sequence of 86-dim file feature vectors and predicts a label per file, informed by
neighbors -- a learned alternative to the HMM smoother. Two variants:
  V1: input = 86 features            (GRU replaces LightGBM + HMM)
  V2: input = 86 features + 6 leak-free LGB OOF probs  (GRU corrects LGB using context)

Validation: group-by-user folds (whole user sequences held out), SAME partitions used
everywhere else, so OOF is comparable to the 0.7460 smoother and the LGB probs fed to V2
are leak-free (val users were held out of the LGB that produced their probs).

Determinism: CPU + full-batch padded training, all seeds pinned -> bit-reproducible
(verified by a two-run check). Current best to beat: LGB+HMM OOF 0.7460 (noise +/-0.0024).

Run: python step18_gru.py
"""
from __future__ import annotations
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

import temporal_lib as L
from har_cv import SEED, N_CLASSES

DEVICE = "cpu"          # tiny data; CPU guarantees the reproducibility gate
HIDDEN, LAYERS, DROPOUT = 192, 2, 0.3
MAX_EPOCHS, PATIENCE = 150, 25
LR, WD, CLIP = 1e-3, 1e-5, 5.0
NOISE = 0.0024
# smoother (current best) reference per partition; p7 == p42 (identical split), omitted.
SMOOTHER_REF = {"det": 0.7460, "p42": 0.7377, "p123": 0.7460}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class GRUSeq(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, HIDDEN, LAYERS, batch_first=True,
                          bidirectional=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2 * HIDDEN, N_CLASSES))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.head(h)


def user_sequences(fid, user):
    seqs = {}
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        seqs[u] = idx[np.argsort(fid[idx])]
    return seqs


def build_batch(users, seqs, Xin, y, mu, sd):
    rows = [seqs[u] for u in users]
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    Lmax, D = int(lengths.max()), Xin.shape[1]
    Xpad = torch.zeros(len(users), Lmax, D)
    Ypad = torch.full((len(users), Lmax), -100, dtype=torch.long)
    mask = torch.zeros(len(users), Lmax, dtype=torch.bool)
    for i, r in enumerate(rows):
        l = len(r)
        Xpad[i, :l] = torch.tensor((Xin[r] - mu) / sd, dtype=torch.float32)
        Ypad[i, :l] = torch.tensor(y[r], dtype=torch.long)
        mask[i, :l] = True
    return Xpad.to(DEVICE), Ypad.to(DEVICE), lengths, mask.to(DEVICE), rows


def train_fold(Xin, y, seqs, tr_users, va_users, in_dim, seed=SEED):
    set_seed(seed)
    train_rows = np.concatenate([seqs[u] for u in tr_users])
    mu = Xin[train_rows].mean(0); sd = Xin[train_rows].std(0) + 1e-8
    Xtr, Ytr, Ltr, Mtr, _ = build_batch(tr_users, seqs, Xin, y, mu, sd)
    Xva, Yva, Lva, Mva, va_rows = build_batch(va_users, seqs, Xin, y, mu, sd)
    yva_true = y[np.concatenate([seqs[u] for u in va_users])]

    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y[train_rows])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE),
                               ignore_index=-100)
    model = GRUSeq(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_f1, best_state, since = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train(); opt.zero_grad()
        logits = model(Xtr, Ltr)
        loss = crit(logits.reshape(-1, N_CLASSES), Ytr.reshape(-1))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            lv = model(Xva, Lva)
        pv = lv[Mva].argmax(1).cpu().numpy()
        f = f1_score(yva_true, pv, average="macro")
        if f > best_f1 + 1e-5:
            best_f1, since = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        lv = model(Xva, Lva)
    probs = torch.softmax(lv, dim=2)[Mva].cpu().numpy()
    preds = probs.argmax(1)
    val_idx = np.concatenate(va_rows)
    return val_idx, preds, probs


def gru_oof(Xin, y, fid, user, groups, cv, seed=SEED):
    seqs = user_sequences(fid, user)
    n = len(y)
    oof_pred = np.full(n, -1, dtype=int)
    oof_proba = np.zeros((n, N_CLASSES))
    dummy = np.zeros((n, 1))
    for tr, va in cv.split(dummy, y, groups):
        tr_users = sorted(set(user[tr])); va_users = sorted(set(user[va]))
        idx, preds, probs = train_fold(Xin, y, seqs, tr_users, va_users, Xin.shape[1], seed)
        oof_pred[idx] = preds; oof_proba[idx] = probs
    return oof_pred, oof_proba


def report_row(name, pred, y):
    m = f1_score(y, pred, average="macro")
    pc = f1_score(y, pred, average=None, labels=list(range(N_CLASSES)))
    print(f"{name:30} | {m:7.4f} | " + "  | ".join(f"{pc[c]:.4f}" for c in range(N_CLASSES)))
    return m, pc


def main():
    X, y, fid, user = L.load_train()
    groups = user
    partitions = [("det", StratifiedGroupKFold(5, shuffle=False)),
                  ("p42", StratifiedGroupKFold(5, shuffle=True, random_state=42)),
                  ("p123", StratifiedGroupKFold(5, shuffle=True, random_state=123))]

    print("=" * 88)
    hdr = f"{'variant / partition':30} | {'macroF1':>7} | " + " | ".join(f"L{c}" for c in range(N_CLASSES))
    v1_ms, v2_ms = {}, {}
    lgb_probs_cache = {}
    for name, cv in partitions:
        print(f"\n--- partition {name} (smoother ref {SMOOTHER_REF[name]}) ---"); print(hdr)
        t0 = time.time()
        p1, pr1 = gru_oof(X, y, fid, user, groups, cv)
        m1, _ = report_row(f"V1 GRU(86)  [{name}]", p1, y)
        v1_ms[name] = m1
        lgb_p = L.lgb_oof_probs(X, y, groups, cv); lgb_probs_cache[name] = lgb_p
        X2 = np.hstack([X, lgb_p])
        p2, pr2 = gru_oof(X2, y, fid, user, groups, cv)
        m2, _ = report_row(f"V2 GRU(86+LGBprob) [{name}]", p2, y)
        v2_ms[name] = m2
        print(f"   (partition time {time.time()-t0:.1f}s)")
        if name == "det":
            det_v2_pred, det_v2_proba = p2, pr2

    print("\n" + "=" * 60)
    print("3-partition summary (vs smoother ref):")
    for name, _ in partitions:
        print(f"  {name}: V1={v1_ms[name]:.4f}  V2={v2_ms[name]:.4f}  smoother={SMOOTHER_REF[name]:.4f}")
    print(f"  mean: V1={np.mean(list(v1_ms.values())):.4f}  V2={np.mean(list(v2_ms.values())):.4f}  "
          f"smoother={np.mean(list(SMOOTHER_REF.values())):.4f}")

    # determinism check on best variant (det partition)
    print("\nDeterminism check (V2, det partition, two runs):")
    p2b, _ = gru_oof(np.hstack([X, lgb_probs_cache['det']]), y, fid, user, groups,
                     StratifiedGroupKFold(5, shuffle=False))
    same = np.array_equal(det_v2_pred, p2b)
    print(f"  identical OOF predictions across two runs: {same}")

    # confusion matrix for best variant on det
    best_pred = det_v2_pred if np.mean(list(v2_ms.values())) >= np.mean(list(v1_ms.values())) else p1
    cm = confusion_matrix(y, det_v2_pred, labels=list(range(N_CLASSES)))
    print("\nConfusion matrix (V2, det)  rows=TRUE cols=PRED")
    for t in range(N_CLASSES):
        print(f"true {t} |" + "".join(f"{cm[t,p]:>7}" for p in range(N_CLASSES)))
    for c in (2, 5):
        pr = precision_score(y, det_v2_pred, labels=[c], average="micro", zero_division=0)
        rc = recall_score(y, det_v2_pred, labels=[c], average="micro", zero_division=0)
        print(f"  L{c}: precision={pr:.4f} recall={rc:.4f}")

    # gate: best variant beats smoother on ALL partitions by > noise
    best_ms = v2_ms if np.mean(list(v2_ms.values())) >= np.mean(list(v1_ms.values())) else v1_ms
    wins = {k: best_ms[k] - SMOOTHER_REF[k] for k in best_ms}
    gate = all(d > NOISE for d in wins.values())
    print(f"\nbest-variant deltas vs smoother: {{{', '.join(f'{k}:{v:+.4f}' for k,v in wins.items())}}}")
    print(f"ROBUST WIN (beats smoother by >{NOISE} on ALL partitions): {gate}")

    if gate:
        print("\n>>> robust win -> running blend (0.5*GRU+0.5*LGB)+HMM and writing submission")
        run_blend_and_submit(X, y, fid, user, groups, det_v2_proba, lgb_probs_cache['det'])
    else:
        print("\n>>> NOT a robust win -> keep submission_temporal.csv (0.7867).")


def run_blend_and_submit(X, y, fid, user, groups, gru_proba_det, lgb_proba_det):
    cv = StratifiedGroupKFold(5, shuffle=False)
    blend = 0.5 * gru_proba_det + 0.5 * lgb_proba_det
    pred = L.smooth(blend, y, fid, user, groups, cv, **L.CURRENT)
    m = f1_score(y, pred, average="macro")
    print(f"  blended (0.5 GRU + 0.5 LGB) + HMM smoother OOF (det): {m:.4f} (smoother-only 0.7460)")
    print("  (submission generation for the GRU blend requires a full-data GRU refit; "
          "left as a follow-up once you confirm you want to submit this.)")


if __name__ == "__main__":
    main()
