"""Rolling robust z detector over the feature pool's stored z (long window, else short)."""
from __future__ import annotations

import math

from . import params as P


def directional(z: float, direction: str) -> float:
    return z if direction == P.UP else -z if direction == P.DOWN else abs(z)


def evaluate(values: dict[str, tuple[float | None, float | None]], n_trades: int, low_snapshots: bool) -> dict | None:
    """values: feature -> (value, z). Returns the unified detector output, or None if nothing is judgeable.

    score = max over judged features of directional z / that feature's threshold (1.0 = at threshold).
    """
    best, hits, judged = -math.inf, [], 0
    for name, direction in P.Z_DIRECTIONS.items():
        value, z = values.get(name, (None, None))
        if z is None or not math.isfinite(z):
            continue
        if name in P.RATIO_FEATURES and n_trades < P.MIN_TRADES_FOR_RATIOS:
            continue
        if name in P.BOOK_FEATURES and low_snapshots:
            continue
        judged += 1
        thr = P.Z_THRESHOLDS.get(name, P.Z_THRESHOLD)
        ratio = directional(z, direction) / thr
        best = max(best, ratio)
        if ratio >= 1.0:
            hits.append({"feature": name, "value": value, "z": z, "threshold": thr, "direction": direction,
                         "ratio": ratio, "type_hint": P.TYPE_HINTS.get(name)})
    if judged == 0:
        return None
    hits.sort(key=lambda h: -h["ratio"])
    return {"detector": "robust_z", "score": best, "threshold": 1.0, "is_anomaly": bool(hits), "evidence": hits}
