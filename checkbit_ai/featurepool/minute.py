"""Read raw data (checkbit_ai/rawdb.py tables) into one MinuteData per minute.

Buckets are Upbit time: minute m covers trade_timestamp / orderbook timestamp in
[m*60s, (m+1)*60s). Coverage comes from collector_sessions (server time).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from .params import MINUTE_MS


@dataclass
class MinuteData:
    minute: int
    covered_ms: int = 0
    prices: np.ndarray = field(default_factory=lambda: np.zeros(0))
    volumes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_buy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # one entry per orderbook snapshot in the minute
    spread_bp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spread_wide: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 1.0 / 0.0 / nan (tick unknown)
    depth15: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oi5: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_trades(self) -> int:
        return len(self.prices)

    @property
    def n_snapshots(self) -> int:
        return len(self.spread_bp)


def _snapshot_stats(asks: list[float], bids: list[float], ask_sz: list[float], bid_sz: list[float]):
    a1, b1 = asks[0], bids[0]
    mid = (a1 + b1) / 2.0
    spread = (a1 - b1) / mid * 1e4 if mid > 0 else math.nan
    # tick = smallest positive gap between adjacent levels on either side
    gaps = [asks[i + 1] - asks[i] for i in range(len(asks) - 1)] + \
           [bids[i] - bids[i + 1] for i in range(len(bids) - 1)]
    gaps = [g for g in gaps if g > 0]
    tick = min(gaps) if gaps else math.nan
    if math.isnan(tick) or math.isnan(spread):
        wide = math.nan
    else:
        wide = 1.0 if (a1 - b1) > tick * (1 + 1e-9) else 0.0
    depth = sum(ask_sz) + sum(bid_sz)
    b5, a5 = sum(bid_sz[:5]), sum(ask_sz[:5])
    oi5 = (b5 - a5) / (b5 + a5) if (b5 + a5) > 0 else math.nan
    return spread, wide, depth, oi5


def coverage_ms(intervals: list[tuple[int, int]], start: int, end: int) -> int:
    """Length of [start, end) covered by the union of intervals."""
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end)
    total, cur_a, cur_b = 0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def load_range(conn: sqlite3.Connection, market: str, m0: int, m1: int, now_ms: int) -> list[MinuteData]:
    """One MinuteData per minute m0..m1 (inclusive)."""
    t0, t1 = m0 * MINUTE_MS, (m1 + 1) * MINUTE_MS
    out = [MinuteData(minute=m) for m in range(m0, m1 + 1)]

    sessions = conn.execute(
        "SELECT connected_at, COALESCE(disconnected_at, ?) FROM collector_sessions"
        " WHERE market = ? AND connected_at < ? AND COALESCE(disconnected_at, ?) > ?",
        (now_ms, market, t1, now_ms, t0)).fetchall()
    for md in out:
        a = md.minute * MINUTE_MS
        md.covered_ms = coverage_ms(sessions, a, a + MINUTE_MS)

    prices: dict[int, list] = {}
    for ts, price, vol, side in conn.execute(
            "SELECT trade_timestamp, trade_price, trade_volume, ask_bid FROM trades"
            " WHERE market = ? AND trade_timestamp >= ? AND trade_timestamp < ?"
            " ORDER BY trade_timestamp, sequential_id", (market, t0, t1)):
        prices.setdefault(ts // MINUTE_MS, []).append((price, vol, side == "BID"))
    for md in out:
        rows = prices.get(md.minute)
        if rows:
            md.prices = np.array([r[0] for r in rows], float)
            md.volumes = np.array([r[1] for r in rows], float)
            md.is_buy = np.array([r[2] for r in rows], bool)

    snaps: dict[int, list] = {}
    cur_id, cur = None, None

    def flush():
        if cur is not None and cur["asks"]:
            snaps.setdefault(cur["ts"] // MINUTE_MS, []).append(
                _snapshot_stats(cur["asks"], cur["bids"], cur["ask_sz"], cur["bid_sz"]))

    for sid, ts, _depth, ap, bp, asz, bsz in conn.execute(
            "SELECT s.id, s.timestamp, u.depth, u.ask_price, u.bid_price, u.ask_size, u.bid_size"
            " FROM orderbook_snapshots s JOIN orderbook_units u ON u.snapshot_id = s.id"
            " WHERE s.market = ? AND s.timestamp >= ? AND s.timestamp < ?"
            " ORDER BY s.timestamp, s.id, u.depth", (market, t0, t1)):
        if sid != cur_id:
            flush()
            cur_id, cur = sid, {"ts": ts, "asks": [], "bids": [], "ask_sz": [], "bid_sz": []}
        cur["asks"].append(ap)
        cur["bids"].append(bp)
        cur["ask_sz"].append(asz)
        cur["bid_sz"].append(bsz)
    flush()
    for md in out:
        rows = snaps.get(md.minute)
        if rows:
            arr = np.array(rows, float)
            md.spread_bp, md.spread_wide, md.depth15, md.oi5 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return out
