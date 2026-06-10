"""New feature family: FFT/SPECTRAL features -- a genuinely different signal axis.

Why this and not "more ensembling": adding more LGB members + TTA, and adding XGB/CAT,
both came back flat/worse on LB -- those levers were correlated draws on the SAME
information (the frozen 86 features). To move the needle we need NEW information.

Families A+B (86 feats) capture distribution shape + time-domain rhythm proxies
(crossing rate, fixed-lag autocorrelation, active fraction, longest run). None of them
directly measure PERIODICITY -- the dominant oscillation frequency and how peaked vs.
flat the power spectrum is. That's exactly what separates locomotion (walking/running:
strong low-frequency gait peak) from static postures (sitting/standing/lying: flat,
DC-dominated spectrum) -- a classic, well-established HAR signal we have not used.

For each of 6 raw channels + 2 magnitude channels (||mean_xyz||, ||std_xyz||), compute a
6-dim FFT summary over the 300-step sequence using only SMOOTH (non-argmax) statistics
-- spectral entropy, centroid, log-energy, and low/mid/high 3-band energy ratios --
chosen to avoid the bin-jitter instability of argmax-based "dominant frequency" stats
(an earlier 6-stat set including dom_freq/dom_power passed the mean-gain bar (+0.0027 >
0.0024) but lost on 1/3 partitions -- exactly the symptom of a noisy, unstable feature).
-> 8 channels * 6 = 48 new features (Family C), concatenated with the frozen 86.

Fair 3-partition gate vs frozen 86-feat LGB+HMM (0.7287 ref). Deterministic (CPU LGB).
Run: python stepFamilyC_spectral.py
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

import temporal_lib as L
import decoder_lib as Dl
from step2_familyB import build_features


def spectral_summary(x):
    """x: (n, T) -> (n, 6) SMOOTH spectral stats (no argmax -- avoids bin-jitter instability):
    entropy, centroid, total_energy(log), low/mid/high 3-band energy ratios."""
    n, T = x.shape
    xc = x - x.mean(1, keepdims=True)
    mag = np.abs(np.fft.rfft(xc, axis=1))[:, 1:]          # drop DC bin -> (n, T//2)
    power = mag ** 2 + 1e-12
    total = power.sum(1, keepdims=True)
    p = power / total
    nf = mag.shape[1]
    freqs = np.arange(1, nf + 1, dtype=float)

    entropy = -(p * np.log(p)).sum(1)
    centroid = (p * freqs).sum(1)
    log_energy = np.log(total[:, 0])
    b1, b2 = nf // 3, 2 * nf // 3
    low_ratio = power[:, :b1].sum(1) / total[:, 0]
    mid_ratio = power[:, b1:b2].sum(1) / total[:, 0]
    high_ratio = power[:, b2:].sum(1) / total[:, 0]
    return np.column_stack([entropy, centroid, log_energy, low_ratio, mid_ratio, high_ratio])


def family_C_features(X_raw):
    """X_raw: (n, 300, 6) -> (n, 48): spectral summary over 6 axes + 2 magnitude channels."""
    chans = [X_raw[:, :, c] for c in range(6)]
    chans.append(np.linalg.norm(X_raw[:, :, 0:3], axis=2))   # ||mean_xyz||
    chans.append(np.linalg.norm(X_raw[:, :, 3:6], axis=2))   # ||std_xyz||
    return np.hstack([spectral_summary(c) for c in chans])


def main():
    X_raw, meta = L.D.load_split("train")
    y = meta["label"].to_numpy(); fid = meta["file_id"].to_numpy(); user = meta["user"].to_numpy()
    groups = user
    X86 = build_features(X_raw)

    print("Computing Family C (spectral/FFT) features...")
    XC = family_C_features(X_raw)
    Xnew = np.hstack([X86, XC])
    print(f"features: {X86.shape[1]} -> {Xnew.shape[1]}  (+{XC.shape[1]} spectral)\n")

    print("Reference: frozen 86-feat LGB+HMM on the 3 distinct partitions (cached)...")
    parts = Dl.get_partitions(X86, y, fid, user)             # [(seed, cv, lgb_probs86, hmm_macro)]
    base_ms = np.array([p[3] for p in parts])
    print(f"  86-feat mean = {base_ms.mean():.4f}  {[round(m, 4) for m in base_ms]}\n")

    print("Candidate: 86+spectral LGB+HMM on the SAME partitions (fresh OOF fit, no cache)...")
    new_ms = []
    for s, cv, _lgb_p86, _bh in parts:
        probs = L.lgb_oof_probs(Xnew, y, groups, cv)
        pred = L.smooth(probs, y, fid, user, groups, cv, **L.CURRENT)
        m = f1_score(y, pred, average="macro")
        new_ms.append(m)
        print(f"  partition {s}: macro-F1 = {m:.4f}")
    new_ms = np.array(new_ms)
    print(f"  86+spectral mean = {new_ms.mean():.4f}  {[round(m, 4) for m in new_ms]}\n")

    print("=" * 60)
    print(f"delta mean: {new_ms.mean() - base_ms.mean():+.4f}   (noise floor {Dl.NOISE})")
    win = Dl.gate(new_ms, base_ms)
    print(f"ROBUST WIN (mean gain > noise AND not worse on any partition): {win}")

    if win:
        print("\n>>> robust win -> writing submission_specC.csv (full refit, reproducibility-checked)")
        Xte_raw, meta_te = L.D.load_split("test")
        fid_te = meta_te["file_id"].to_numpy(); user_te = meta_te["user"].to_numpy()
        X86_te = build_features(Xte_raw)
        XC_te = family_C_features(Xte_raw)
        Xte = np.hstack([X86_te, XC_te])
        L.write_submission(Xnew, y, fid, user, L.CURRENT,
                           "/root/dm-assignment3/submission_specC.csv", X_test=Xte)
    else:
        print("\n>>> not a robust win -> drop spectral family, keep current best (aug ensemble, LB 0.7958).")


if __name__ == "__main__":
    main()
