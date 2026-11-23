"""process_minutes(): the one function realtime and replay both run.

For each chunk of target minutes:
1. read raw data for [chunk start - LOOKBACK_MINUTES, chunk end] from raw.sqlite3,
2. compute raw features for the targets and store them,
3. normalize each target against the *stored* raw values of earlier minutes.

Realtime calls it for the minutes finalized since the last run, so its output
equals a replay of the same minutes, given the earlier minutes were computed.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from . import params as P
from . import store
from .definitions import FEATURE_NAMES, MAD_FLOORS, compute
from .minute import load_range
from .normalize import Normalizer

LOG = logging.getLogger("checkbit.featurepool")
CHUNK_MINUTES = 720


def now_ms() -> int:
    return int(time.time() * 1000)


def last_final_minute(now: int) -> int:
    """Latest minute m with now >= (m+1)*60s + grace."""
    return (now - P.GRACE_MS) // P.MINUTE_MS - 1


def open_raw(path: Path) -> sqlite3.Connection:
    # read-only; the collector's writer owns writes
    return sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, timeout=10)


def first_session_minute(raw: sqlite3.Connection, market: str) -> int | None:
    v = raw.execute("SELECT MIN(connected_at) FROM collector_sessions WHERE market = ?", (market,)).fetchone()[0]
    return None if v is None else v // P.MINUTE_MS


def process_minutes(raw: sqlite3.Connection, feat: sqlite3.Connection, market: str, m0: int, m1: int, *,
                    source: str, now: int | None = None, orderbook_interval_ms: int = 1000,
                    normalizer: Normalizer | None = None) -> int:
    """Compute and store features for minutes m0..m1 (clamped to finalized minutes). Returns #minutes."""
    now = now_ms() if now is None else now
    m1 = min(m1, last_final_minute(now))
    if m1 < m0:
        return 0
    norm = normalizer or Normalizer()
    done = 0
    for c0 in range(m0, m1 + 1, CHUNK_MINUTES):
        c1 = min(c0 + CHUNK_MINUTES - 1, m1)
        lo = c0 - P.LOOKBACK_MINUTES
        minutes = load_range(raw, market, lo, c1, now)
        metas, values = compute(minutes, first_target=c0 - lo, orderbook_interval_ms=orderbook_interval_ms)
        with feat:
            store.write_raw(feat, market, metas, values, P.DEFS_VERSION, source, now)
        # normalize against stored history (+ the values just written)
        h0 = c0 - norm.lookback_minutes
        series = store.load_series(feat, market, h0, c1, FEATURE_NAMES)
        rows = []
        for minute in range(c0, c1 + 1):
            i = minute - h0
            for name in FEATURE_NAMES:
                zs, zl, z = norm.z(series[name], i, MAD_FLOORS[name])
                rows.append((zs, zl, z, market, minute, name))
        with feat:
            store.write_z(feat, rows)
        done += c1 - c0 + 1
    return done
