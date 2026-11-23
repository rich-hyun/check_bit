"""Group per-minute detector hits into candidate intervals for the agent.

Rules (params): hits at most MERGE_GAP hit-free minutes apart belong to one
candidate; it closes after MERGE_GAP+1 hit-free minutes, or when it spans
MAX_CANDIDATE_MINUTES (a new one starts). Tier: strong if >= STRONG_MIN_FAMILIES
detector families hit, or any hit reached STRONG_RATIO x its threshold; else ambiguous.

step() reads only stored detector_hits, so realtime and a rebuild give the same result.
"""
from __future__ import annotations

import json
import sqlite3

from . import params as P
from . import store


def minute_summary(rows: list[tuple]) -> dict:
    """rows from store.hits_for_minute -> what the candidate builder needs."""
    s = {"hit": False, "ratios": {}, "types": set(), "lstm_available": False}
    for detector, score, threshold, is_anomaly, evidence in rows:
        if detector == "lstm":
            s["lstm_available"] = True
        if not is_anomaly:
            continue
        s["hit"] = True
        s["ratios"][detector] = score / threshold if threshold else score
        if detector == "jump":
            s["types"].add(P.JUMP_TYPE)
        elif detector == "lstm":
            s["types"].add(P.LSTM_TYPE)
        else:
            for h in json.loads(evidence):
                if h.get("type_hint"):
                    s["types"].add(h["type_hint"])
    return s


def _tier(detectors: dict, max_ratio: float) -> str:
    families = sum(1 for d in detectors.values() if d["hit_minutes"] > 0)
    return "strong" if families >= P.STRONG_MIN_FAMILIES or max_ratio >= P.STRONG_RATIO else "ambiguous"


def _open(conn: sqlite3.Connection, market: str):
    return conn.execute("SELECT id, start_minute, end_minute, peak_minute, detectors, types, max_ratio,"
                        " hit_minutes, lstm_minutes FROM candidates WHERE market = ? AND status = 'open'",
                        (market,)).fetchone()


def _close(conn: sqlite3.Connection, cid: int, minute: int) -> None:
    conn.execute("UPDATE candidates SET status = 'closed', closed_at_minute = ? WHERE id = ?", (minute, cid))


def step(conn: sqlite3.Connection, market: str, minute: int) -> None:
    """Advance the candidate state by one minute (minutes must be fed in order)."""
    s = minute_summary(store.hits_for_minute(conn, market, minute))
    cur = _open(conn, market)
    if cur is not None and minute - cur[1] >= P.MAX_CANDIDATE_MINUTES:
        _close(conn, cur[0], minute)
        cur = None
    if not s["hit"]:
        if cur is not None and minute - cur[2] > P.MERGE_GAP:
            _close(conn, cur[0], minute)
        return
    if cur is not None and minute - cur[2] > P.MERGE_GAP + 1:   # too far from the last hit
        _close(conn, cur[0], minute)
        cur = None
    peak_ratio = max(s["ratios"].values())
    if cur is None:
        dets = {d: {"max_ratio": r, "hit_minutes": 1} for d, r in s["ratios"].items()}
        conn.execute(
            "INSERT INTO candidates (market, start_minute, end_minute, peak_minute, status, tier, detectors, types,"
            " max_ratio, hit_minutes, lstm_minutes, version) VALUES (?,?,?,?,'open',?,?,?,?,1,?,?)",
            (market, minute, minute, minute, _tier(dets, peak_ratio), json.dumps(dets, ensure_ascii=False),
             json.dumps(sorted(s["types"]), ensure_ascii=False), peak_ratio, int(s["lstm_available"]),
             P.TRIGGER_VERSION))
        return
    cid, start, _end, peak, dets_j, types_j, max_ratio, hit_minutes, lstm_minutes = cur
    dets = json.loads(dets_j)
    for d, r in s["ratios"].items():
        e = dets.setdefault(d, {"max_ratio": r, "hit_minutes": 0})
        e["max_ratio"] = max(e["max_ratio"], r)
        e["hit_minutes"] += 1
    if peak_ratio > max_ratio:
        max_ratio, peak = peak_ratio, minute
    types = sorted(set(json.loads(types_j)) | s["types"])
    conn.execute("UPDATE candidates SET end_minute = ?, peak_minute = ?, tier = ?, detectors = ?, types = ?,"
                 " max_ratio = ?, hit_minutes = ?, lstm_minutes = ? WHERE id = ?",
                 (minute, peak, _tier(dets, max_ratio), json.dumps(dets, ensure_ascii=False),
                  json.dumps(types, ensure_ascii=False), max_ratio, hit_minutes + 1,
                  lstm_minutes + int(s["lstm_available"]), cid))


def rebuild(conn: sqlite3.Connection, market: str, until_minute: int) -> int:
    """Recreate all candidates from stored detector_hits; keeps agent_status of identical intervals."""
    keep = {(a, b): st for a, b, st in conn.execute(
        "SELECT start_minute, end_minute, agent_status FROM candidates WHERE market = ? AND status = 'closed'",
        (market,))}
    conn.execute("DELETE FROM candidates WHERE market = ?", (market,))
    first = conn.execute("SELECT MIN(minute) FROM detector_hits WHERE market = ?", (market,)).fetchone()[0]
    if first is None:
        return 0
    for m in range(first, until_minute + 1):
        step(conn, market, m)
    for (a, b), st in keep.items():
        conn.execute("UPDATE candidates SET agent_status = ? WHERE market = ? AND start_minute = ?"
                     " AND end_minute = ? AND status = 'closed'", (st, market, a, b))
    return conn.execute("SELECT COUNT(*) FROM candidates WHERE market = ?", (market,)).fetchone()[0]
