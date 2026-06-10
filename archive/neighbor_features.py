"""Idea 5: neighbor / lag / delta context features for LightGBM.

For each file (within its user, in recording order = file_id) augment the 86 within-file
features with:
  - lag1  : previous file's 86 features        (neighbor context)
  - lead1 : next file's 86 features            (neighbor context)
  - dprev : current - previous  (86)           (transition / change signal)
  - dnext : next - current      (86)           (transition / change signal)
  - position: index-in-session, normalized position, session length (3)
Total = 86*5 + 3 = 433.

No leakage: these are FEATURES (always available at test time too), not labels. Session
boundaries are padded with the file's own vector, so deltas are 0 there (no fake jump).
This lets a sample-efficient tree (11k rows) learn context-dependent boundaries -- e.g.
"a borderline-std run of several files = an L2 run; a lone borderline file = L1 noise" --
which the frozen per-file model and the data-starved GRU could not.
"""
from __future__ import annotations
import numpy as np


def build_neighbor_features(X, fid, user):
    n, D = X.shape
    lag1 = X.copy()
    lead1 = X.copy()
    dprev = np.zeros_like(X)
    dnext = np.zeros_like(X)
    pos = np.zeros((n, 3))
    for u in np.unique(user):
        idx = np.where(user == u)[0]
        order = idx[np.argsort(fid[idx])]
        Xu = X[order]
        L = len(order)
        lp = np.vstack([Xu[0:1], Xu[:-1]])      # prev (pad first with self)
        ln = np.vstack([Xu[1:], Xu[-1:]])       # next (pad last with self)
        lag1[order] = lp
        lead1[order] = ln
        dprev[order] = Xu - lp
        dnext[order] = ln - Xu
        pos[order, 0] = np.arange(L)
        pos[order, 1] = np.arange(L) / max(1, L - 1)
        pos[order, 2] = L
    return np.hstack([X, lag1, lead1, dprev, dnext, pos])
