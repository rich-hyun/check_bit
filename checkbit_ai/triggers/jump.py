"""Lee & Mykland (2008) jump test on 1-minute log returns.

L_t = r_t / σ̂_t with σ̂_t² = bipower_var_t from the feature pool (current return
excluded). Under no jump, c·L_t is approximately standard normal, c = √(2/π).

Decision (realtime): reject when (|L_t| − C_n) / S_n > β*, β* = −log(−log(1 − α)),
    C_n = √(2 log n)/c − (log π + log log n) / (2c √(2 log n)),   S_n = 1 / (c √(2 log n)),
which bounds the probability of any false jump among n tests by α (here: per day).
Replay additionally reports Benjamini-Hochberg over each day's p-values.
"""
from __future__ import annotations

import math

import numpy as np

from . import params as P

C = math.sqrt(2.0 / math.pi)


def gumbel_constants(n: int) -> tuple[float, float]:
    s = math.sqrt(2.0 * math.log(n))
    cn = s / C - (math.log(math.pi) + math.log(math.log(n))) / (2.0 * C * s)
    sn = 1.0 / (C * s)
    return cn, sn


def beta_star(alpha: float) -> float:
    return -math.log(-math.log(1.0 - alpha))


CN, SN = gumbel_constants(P.JUMP_N)
BETA = beta_star(P.JUMP_ALPHA)


def evaluate(r: float | None, bipower: float | None) -> dict | None:
    """Unified detector output, or None when the test cannot run (gap, warm-up)."""
    if r is None or bipower is None or not math.isfinite(r) or not math.isfinite(bipower) or bipower <= 0:
        return None
    L = r / math.sqrt(bipower)
    g = (abs(L) - CN) / SN
    return {"detector": "jump", "score": g, "threshold": BETA, "is_anomaly": g > BETA,
            "evidence": [{"feature": "price_return_1m", "value": r, "L": L, "cL": C * L,
                          "p_value": math.erfc(abs(C * L) / math.sqrt(2.0)),
                          "direction": "up" if r > 0 else "down"}]}


def bh_reject(pvalues: np.ndarray, q: float = P.BH_Q) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of rejected hypotheses."""
    p = np.asarray(pvalues, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(m, bool)
    mask[order[:k]] = True
    return mask
