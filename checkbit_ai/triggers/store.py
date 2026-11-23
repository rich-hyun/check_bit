"""Trigger tables, kept in features.sqlite3 next to the feature pool."""
from __future__ import annotations

import json
import math
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS detector_hits (
    market     TEXT    NOT NULL,
    minute     INTEGER NOT NULL,
    detector   TEXT    NOT NULL,          -- jump | robust_z | lstm
    score      REAL    NOT NULL,
    threshold  REAL    NOT NULL,
    is_anomaly INTEGER NOT NULL,
    evidence   TEXT    NOT NULL,          -- JSON list
    version    TEXT    NOT NULL,
    PRIMARY KEY (market, minute, detector)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY,
    market          TEXT    NOT NULL,
    start_minute    INTEGER NOT NULL,
    end_minute      INTEGER NOT NULL,     -- last hit minute
    peak_minute     INTEGER NOT NULL,
    status          TEXT    NOT NULL,     -- open | closed
    tier            TEXT    NOT NULL,     -- strong | ambiguous
    detectors       TEXT    NOT NULL,     -- JSON {detector: {max_ratio, hit_minutes}}
    types           TEXT    NOT NULL,     -- JSON list of L2 type hints
    max_ratio       REAL    NOT NULL,
    hit_minutes     INTEGER NOT NULL,
    lstm_minutes    INTEGER NOT NULL,     -- hit minutes that had an LSTM result
    agent_status    TEXT    NOT NULL DEFAULT 'pending',  -- pending | done | skipped (agent queue)
    version         TEXT    NOT NULL,
    closed_at_minute INTEGER
);
CREATE INDEX IF NOT EXISTS idx_candidates_market ON candidates(market, start_minute);
CREATE INDEX IF NOT EXISTS idx_candidates_queue ON candidates(status, agent_status);

CREATE TABLE IF NOT EXISTS trigger_state (
    market      TEXT PRIMARY KEY,
    last_minute INTEGER NOT NULL
);
"""


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _f(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def write_hit(conn: sqlite3.Connection, market: str, minute: int, out: dict, version: str) -> None:
    conn.execute("INSERT OR REPLACE INTO detector_hits VALUES (?,?,?,?,?,?,?,?)",
                 (market, minute, out["detector"], out["score"], out["threshold"], int(out["is_anomaly"]),
                  json.dumps(out["evidence"], ensure_ascii=False, default=_f), version))


def hits_for_minute(conn: sqlite3.Connection, market: str, minute: int) -> list[tuple]:
    return conn.execute("SELECT detector, score, threshold, is_anomaly, evidence FROM detector_hits"
                        " WHERE market = ? AND minute = ? ORDER BY detector", (market, minute)).fetchall()


def get_state(conn: sqlite3.Connection, market: str) -> int | None:
    r = conn.execute("SELECT last_minute FROM trigger_state WHERE market = ?", (market,)).fetchone()
    return None if r is None else r[0]


def set_state(conn: sqlite3.Connection, market: str, minute: int | None) -> None:
    if minute is None:
        conn.execute("DELETE FROM trigger_state WHERE market = ?", (market,))
    else:
        conn.execute("INSERT OR REPLACE INTO trigger_state VALUES (?, ?)", (market, minute))
