"""EXPERIMENT C -- Transformer across files (frozen LGB emissions).

Learnable positional encoding + 2-layer bidirectional Transformer encoder (nhead=2,
ff=32) over the user's file sequence. Input = 6-dim LGB probs. Same gate as A/B.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score
import decoder_lib as Dl
import temporal_lib as L


def main():
    X, y, fid, user = L.load_train()
    parts = Dl.get_partitions(X, y, fid, user)
    hmm = [p[3] for p in parts]
    print(f"HMM reference (3 distinct partitions): {[round(h,4) for h in hmm]} mean={np.mean(hmm):.4f}\n")

    macros, p0, _ = Dl.run_decoder(Dl.TransformerSeq, False, Dl.input6, parts, X, y, fid, user)
    pc = f1_score(y, p0, average=None, labels=list(range(Dl.N_CLASSES)))
    g = Dl.gate(macros, hmm)

    print("=" * 84)
    print(f"{'variant':24} | {'mean':>7} | {'p0':>7} {'p1':>7} {'p2':>7} | gate")
    print(f"{'HMM smoother':24} | {np.mean(hmm):7.4f} | {hmm[0]:7.4f} {hmm[1]:7.4f} {hmm[2]:7.4f} |  ref")
    print(f"{'C: Transformer (6-dim)':24} | {np.mean(macros):7.4f} | "
          f"{macros[0]:7.4f} {macros[1]:7.4f} {macros[2]:7.4f} | {g}")
    print("  per-class: " + " ".join(f"L{c}={pc[c]:.3f}" for c in range(Dl.N_CLASSES)))
    Dl.diagnostics(y, p0, "Transformer [part0]")
    print(f"\nROBUST WIN: {g}")


if __name__ == "__main__":
    main()
