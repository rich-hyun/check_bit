"""features.sqlite3: per-minute quality info and long-format feature values."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_minutes (
    market          TEXT    NOT NULL,
    minute          INTEGER NOT NULL,     -- epoch minute (UTC), Upbit-time bucket
    covered_ms      INTEGER NOT NULL,     -- covered by collector_sessions
    is_gap          INTEGER NOT NULL,
    n_trades        INTEGER NOT NULL,
    n_snapshots     INTEGER NOT NULL,
    low_snapshots   INTEGER NOT NULL,     -- secondary check (values still computed)
    roll_cov_nonneg INTEGER NOT NULL,     -- roll_measure NaN because Cov >= 0
    defs_version    TEXT    NOT NULL,
    source          TEXT    NOT NULL,     -- realtime | replay
    computed_at     INTEGER NOT NULL,     -- server epoch ms
    PRIMARY KEY (market, minute)
);
CREATE TABLE IF NOT EXISTS feature_values (
    market  TEXT    NOT NULL,
    minute  INTEGER NOT NULL,
    feature TEXT    NOT NULL,
    value   REAL,                         -- NULL = NaN
    z_short REAL,
    z_long  REAL,
    z       REAL,                         -- z_long, else z_short
    PRIMARY KEY (market, minute, feature)
) WITHOUT ROWID;
"""


def connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _db(v: float):
    return None if v is None or not math.isfinite(v) else float(v)


def write_raw(conn: sqlite3.Connection, market: str, metas, values: dict[str, np.ndarray],
              defs_version: str, source: str, now_ms: int) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO feature_minutes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(market, m.minute, m.covered_ms, int(m.is_gap), m.n_trades, m.n_snapshots, int(m.low_snapshots),
          int(m.roll_cov_nonneg), defs_version, source, now_ms) for m in metas])
    rows = []
    for name, arr in values.items():
        for k, m in enumerate(metas):
            rows.append((market, m.minute, name, _db(arr[k]), None, None, None))
    conn.executemany("INSERT OR REPLACE INTO feature_values VALUES (?,?,?,?,?,?,?)", rows)


def load_series(conn: sqlite3.Connection, market: str, m0: int, m1: int, names: list[str]) -> dict[str, np.ndarray]:
    """Raw values for minutes m0..m1, aligned by minute (missing/NULL -> NaN)."""
    n = m1 - m0 + 1
    out = {name: np.full(n, np.nan) for name in names}
    for minute, feature, value in conn.execute(
            "SELECT minute, feature, value FROM feature_values WHERE market = ? AND minute BETWEEN ? AND ?",
            (market, m0, m1)):
        arr = out.get(feature)
        if arr is not None and value is not None:
            arr[minute - m0] = value
    return out


def write_z(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: (z_short, z_long, z, market, minute, feature)"""
    conn.executemany("UPDATE feature_values SET z_short = ?, z_long = ?, z = ?"
                     " WHERE market = ? AND minute = ? AND feature = ?",
                     [(_db(a), _db(b), _db(c), mk, mi, f) for a, b, c, mk, mi, f in rows])


def last_minute(conn: sqlite3.Connection, market: str) -> int | None:
    return conn.execute("SELECT MAX(minute) FROM feature_minutes WHERE market = ?", (market,)).fetchone()[0]
