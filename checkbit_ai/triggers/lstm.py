"""Wrap the teammate's LSTM AE results (analyses table in checkbit.sqlite3) as a detector.

The model itself is untouched. A result on a collection-gap minute is ignored: the
LSTM pipeline fills short gaps with zero-volume bars, so a hit there is likely an
artifact of missing data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def load(analyses_db: Path | None, market: str, m0: int, m1: int) -> dict[int, tuple]:
    """minute -> (score, threshold, is_anomaly, model_version); the latest row per minute."""
    if analyses_db is None or not Path(analyses_db).exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{Path(analyses_db)}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return {}
    try:
        rows = conn.execute("SELECT minute, score, threshold, is_anomaly, model_version FROM analyses"
                            " WHERE market = ? AND minute BETWEEN ? AND ? ORDER BY id", (market, m0, m1)).fetchall()
    except sqlite3.Error:  # table not created yet
        return {}
    finally:
        conn.close()
    return {r[0]: r[1:] for r in rows}


def evaluate(row: tuple | None, is_gap: bool) -> dict | None:
    if row is None or is_gap:
        return None
    score, threshold, _is_anomaly, version = row
    return {"detector": "lstm", "score": float(score), "threshold": float(threshold),
            "is_anomaly": bool(score > threshold),
            "evidence": [{"model_version": version, "score": float(score), "threshold": float(threshold)}]}
