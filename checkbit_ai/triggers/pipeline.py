"""process_triggers(): evaluate the three detectors for stored feature minutes,
store their unified outputs in detector_hits, and advance candidate intervals.

Realtime (FeatureService) calls it right after the feature pool, LSTM_LAG_MINUTES
behind; replay recomputes detector_hits for a range and then rebuilds candidates.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import candidates, jump, lstm, robust_z
from . import params as P
from . import store

NEEDED = sorted(set(P.Z_DIRECTIONS) | {"price_return_1m", "bipower_var"})


def _load(feat: sqlite3.Connection, market: str, m0: int, m1: int):
    meta = {m: (bool(g), n, bool(low)) for m, g, n, low in feat.execute(
        "SELECT minute, is_gap, n_trades, low_snapshots FROM feature_minutes"
        " WHERE market = ? AND minute BETWEEN ? AND ?", (market, m0, m1))}
    vals: dict[int, dict] = {}
    q = ("SELECT minute, feature, value, z FROM feature_values WHERE market = ? AND minute BETWEEN ? AND ?"
         f" AND feature IN ({','.join('?' * len(NEEDED))})")
    for m, f, v, z in feat.execute(q, (market, m0, m1, *NEEDED)):
        vals.setdefault(m, {})[f] = (v, z)
    return meta, vals


def process_triggers(feat: sqlite3.Connection, analyses_db: Path | None, market: str, m0: int, m1: int, *,
                     update_candidates: bool = True) -> int:
    """Evaluate minutes m0..m1 in order. Returns the number of minutes processed."""
    if m1 < m0:
        return 0
    store.ensure(feat)
    meta, vals = _load(feat, market, m0, m1)
    lstm_rows = lstm.load(analyses_db, market, m0, m1) if P.LSTM_ENABLED else {}
    with feat:
        feat.execute("DELETE FROM detector_hits WHERE market = ? AND minute BETWEEN ? AND ?", (market, m0, m1))
        for m in range(m0, m1 + 1):
            is_gap, n_trades, low = meta.get(m, (True, 0, False))   # no feature row: treat as gap
            v = vals.get(m, {})
            outs = []
            if not is_gap:
                outs.append(jump.evaluate(v.get("price_return_1m", (None, None))[0],
                                          v.get("bipower_var", (None, None))[0]))
                outs.append(robust_z.evaluate(v, n_trades, low))
            if P.LSTM_ENABLED:
                outs.append(lstm.evaluate(lstm_rows.get(m), is_gap))
            for out in outs:
                if out is not None:
                    store.write_hit(feat, market, m, out, P.TRIGGER_VERSION)
            if update_candidates:
                candidates.step(feat, market, m)
            store.set_state(feat, market, m)
    return m1 - m0 + 1


def trigger_range(feat: sqlite3.Connection, market: str) -> tuple[int, int] | None:
    """Next minutes the realtime service should evaluate: after the last evaluated one,
    up to the last finalized feature minute (minus LSTM_LAG_MINUTES when LSTM is on)."""
    store.ensure(feat)
    last_feat = feat.execute("SELECT MAX(minute) FROM feature_minutes WHERE market = ?", (market,)).fetchone()[0]
    if last_feat is None:
        return None
    end = last_feat - P.lag_minutes()
    last = store.get_state(feat, market)
    if last is None:
        first = feat.execute("SELECT MIN(minute) FROM feature_minutes WHERE market = ?", (market,)).fetchone()[0]
        start = first
    else:
        start = last + 1
    return (start, end) if end >= start else None
