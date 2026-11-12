"""Statistical comparison baseline (no learning beyond the training-set scaler).

score = mean of squared standardized feature values over the window, i.e. the
average squared z-distance from the training mean (a diagonal-Gaussian
deviation score). It ignores temporal structure, so comparing it with the LSTM
AutoEncoder shows what the sequence model adds.
"""
from __future__ import annotations

import numpy as np


def baseline_scores(X_scaled_windows: np.ndarray, last_steps: int | None = None) -> np.ndarray:
    if len(X_scaled_windows) == 0:
        return np.zeros(0)
    X = X_scaled_windows[:, -last_steps:] if last_steps else X_scaled_windows
    return np.mean(X.astype(float) ** 2, axis=(1, 2))


def roc_auc(neg: np.ndarray, pos: np.ndarray) -> float | None:
    """Rank-based ROC-AUC (Mann-Whitney U); ties counted as 0.5."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return None
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])
