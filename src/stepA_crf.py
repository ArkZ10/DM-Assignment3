"""EXPERIMENT A -- BiGRU + neural CRF over file sequences (frozen LGB emissions).

Inputs: (a) 6-dim LGB probs, (b) 12-dim = probs + onehot(argmax). BiGRU(32,1) -> linear
unary potentials -> CRF joint decode. Loss = CRF-NLL + class-weighted CE on unaries.
Gate: mean across 3 distinct partitions > HMM mean + 0.0024 AND not worse on any.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score
import decoder_lib as Dl
import temporal_lib as L


def main():
    X, y, fid, user = L.load_train()
    print("Building 3 distinct partitions + recomputing HMM reference...")
    parts = Dl.get_partitions(X, y, fid, user)
    hmm = [p[3] for p in parts]
    print(f"seqs/fold sanity (partition0): val users per fold = "
          f"{[len(np.unique(user[va])) for _, va in Dl.GroupPartition(0).split(X,y,user)]}")
    print(f"HMM reference on distinct partitions {Dl.PART_SEEDS}: "
          f"{[round(h,4) for h in hmm]}  mean={np.mean(hmm):.4f}\n")

    rows = []
    for tag, in_fn in [("A: BiGRU+CRF (6-dim)", Dl.input6),
                       ("A: BiGRU+CRF (12-dim)", Dl.input12)]:
        macros, p0, _ = Dl.run_decoder(Dl.BiGRUCRF, True, in_fn, parts, X, y, fid, user)
        pc = f1_score(y, p0, average=None, labels=list(range(Dl.N_CLASSES)))
        rows.append((tag, macros, pc, p0))
        print(f"{tag}: per-partition {[round(m,4) for m in macros]} mean={np.mean(macros):.4f}")
        Dl.diagnostics(y, p0, tag + " [part0]")

    print("\n" + "=" * 84)
    hdr = f"{'variant':24} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate"
    print(hdr); print("-" * len(hdr))
    print(f"{'HMM smoother':24} | {np.mean(hmm):7.4f} | "
          f"{hmm[0]:7.4f} {hmm[1]:7.4f} {hmm[2]:7.4f} |  ref")
    for tag, macros, pc, p0 in rows:
        g = Dl.gate(macros, hmm)
        print(f"{tag:24} | {np.mean(macros):7.4f} | "
              f"{macros[0]:7.4f} {macros[1]:7.4f} {macros[2]:7.4f} | {g}")
        print(f"  {'':22}   per-class: " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(Dl.N_CLASSES)))


if __name__ == "__main__":
    main()
