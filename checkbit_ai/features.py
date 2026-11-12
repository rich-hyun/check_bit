"""Preprocessing: raw Upbit trade/orderbook CSV rows -> 1-minute bars -> features.

The collector writes one CSV row per *trade*. Each row also carries the most
recent level-1 orderbook snapshot that had been received when the trade
arrived (bid/ask price & size, total bid/ask size). Only these fields are used;
nothing that is absent from the collected data is invented.

All computations are causal: the feature value at minute t only uses data from
minutes <= t. Scaling parameters are fitted on the training split only.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator

import numpy as np

CSV_HEADER = [
    "timestamp", "market", "trade_price", "trade_volume", "ask_bid",
    "bid_price", "ask_price", "bid_size", "ask_size",
    "total_bid_size", "total_ask_size", "trade_timestamp", "sequential_id",
]
LEGACY_HEADER = CSV_HEADER[:11]

# Model input features (order matters: it is saved with the model).
FEATURES = [
    "log_return",          # ln(close_t / close_{t-1})
    "hl_range",            # ln(high / low) within the minute (intra-minute volatility)
    "rv_10",               # std of the last 10 one-minute log returns (rolling volatility)
    "d_log_volume",        # change of log1p(volume in mBTC) vs previous minute
    "log_value",           # log1p(traded value in million KRW)
    "d_log_value",         # change of log_value vs previous minute
    "log_trade_count",     # log1p(number of trades in the minute)
    "buy_sell_imbalance",  # (buy_vol - sell_vol) / (buy_vol + sell_vol), taker side
    "spread_bps",          # (ask - bid) / mid * 1e4 at the last snapshot of the minute
    "l1_imbalance",        # (bid_size - ask_size) / (bid_size + ask_size), best level
    "depth_imbalance",     # (total_bid - total_ask) / (total_bid + total_ask)
]
FEATURE_LABELS = {
    "log_return": "1분 수익률",
    "hl_range": "1분 고저 변동폭",
    "rv_10": "10분 실현 변동성",
    "d_log_volume": "거래량 변화",
    "log_value": "거래대금 규모",
    "d_log_value": "거래대금 변화",
    "log_trade_count": "체결 건수",
    "buy_sell_imbalance": "매수·매도 체결 불균형",
    "spread_bps": "호가 스프레드",
    "l1_imbalance": "최우선 호가 잔량 불균형",
    "depth_imbalance": "전체 호가 잔량 불균형",
}
RV_WINDOW = 10
MAX_FILL_GAP = 5  # up to this many missing minutes are filled; longer gaps split segments


# --------------------------------------------------------------------------
# Raw trade rows
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Trade:
    ts: float              # epoch seconds (UTC)
    price: float
    volume: float
    side: str              # "BID" (buyer-initiated) or "ASK" (seller-initiated)
    bid_price: float = math.nan
    ask_price: float = math.nan
    bid_size: float = math.nan
    ask_size: float = math.nan
    total_bid: float = math.nan
    total_ask: float = math.nan
    seq: str = ""

    @property
    def has_book(self) -> bool:
        return not math.isnan(self.bid_price)


@dataclass
class CleanStats:
    rows: int = 0
    invalid: int = 0
    duplicates: int = 0
    other_market: int = 0
    missing_book: int = 0
    reasons: dict = field(default_factory=dict)

    def bad(self, reason: str) -> None:
        self.invalid += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {"rows": self.rows, "invalid": self.invalid, "duplicates": self.duplicates,
                "other_market": self.other_market, "missing_orderbook": self.missing_book,
                "invalid_reasons": dict(self.reasons)}


def _num(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def parse_timestamp(row: dict) -> float:
    """Prefer Upbit's trade_timestamp (ms, UTC); legacy rows only have the
    receive time written with datetime.now() (naive, server local time)."""
    ms = _num(row.get("trade_timestamp"))
    if ms == ms and ms > 1e12:
        return ms / 1000.0
    raw = (row.get("timestamp") or "").strip()
    if not raw:
        return math.nan
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return math.nan
    return dt.timestamp() if dt.tzinfo else dt.astimezone().timestamp()


def parse_trade_row(row: dict, market: str, stats: CleanStats) -> Trade | None:
    stats.rows += 1
    if (row.get("market") or "").strip() != market:
        stats.other_market += 1
        return None
    ts = parse_timestamp(row)
    if ts != ts:
        stats.bad("timestamp")
        return None
    price, volume = _num(row.get("trade_price")), _num(row.get("trade_volume"))
    if not (price > 0):
        stats.bad("price")
        return None
    if not (volume > 0):
        stats.bad("volume")
        return None
    side = (row.get("ask_bid") or "").strip().upper()
    if side not in ("ASK", "BID"):
        stats.bad("ask_bid")
        return None
    t = Trade(ts=ts, price=price, volume=volume, side=side,
              seq=str(row.get("sequential_id") or "").strip())
    bp, ap = _num(row.get("bid_price")), _num(row.get("ask_price"))
    bs, as_ = _num(row.get("bid_size")), _num(row.get("ask_size"))
    tb, ta = _num(row.get("total_bid_size")), _num(row.get("total_ask_size"))
    # 0 means "no orderbook snapshot received yet" in the collector -> missing.
    if bp > 0 and ap > bp and bs >= 0 and as_ >= 0 and tb > 0 and ta > 0:
        t.bid_price, t.ask_price, t.bid_size, t.ask_size = bp, ap, bs, as_
        t.total_bid, t.total_ask = tb, ta
    else:
        stats.missing_book += 1
    return t


def iter_csv_rows(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if None in row:  # more columns than header -> corrupted line
                row = {k: v for k, v in row.items() if k is not None}
                row["__extra__"] = True
            yield row


def load_trades(paths: Iterable[str | os.PathLike], market: str = "KRW-BTC") -> tuple[list[Trade], CleanStats]:
    """Read one or more CSV files, validate, de-duplicate and sort by time."""
    stats = CleanStats()
    trades: list[Trade] = []
    seen: set = set()
    for path in paths:
        for row in iter_csv_rows(path):
            if row.get("__extra__"):
                stats.rows += 1
                stats.bad("column_count")
                continue
            t = parse_trade_row(row, market, stats)
            if t is None:
                continue
            # Upbit sequential_id is unique per trade. Legacy rows have none, so
            # only exact duplicate rows are removed for them.
            key = ("s", t.seq) if t.seq else ("r", t.ts, t.price, t.volume, t.side, t.bid_price,
                                               t.ask_price, t.bid_size, t.ask_size)
            if key in seen:
                stats.duplicates += 1
                continue
            seen.add(key)
            trades.append(t)
    trades.sort(key=lambda t: t.ts)
    return trades, stats


# --------------------------------------------------------------------------
# 1-minute bars
# --------------------------------------------------------------------------
@dataclass(slots=True)
class MinuteBar:
    minute: int            # epoch minute (UTC): floor(ts / 60)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    value: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    bid_price: float = math.nan
    ask_price: float = math.nan
    bid_size: float = math.nan
    ask_size: float = math.nan
    total_bid: float = math.nan
    total_ask: float = math.nan
    filled: bool = False   # True if synthesized to fill a short gap (no trades)

    @classmethod
    def start(cls, t: Trade) -> "MinuteBar":
        bar = cls(minute=int(t.ts // 60), open=t.price, high=t.price, low=t.price, close=t.price)
        bar.add(t)
        return bar

    def add(self, t: Trade) -> None:
        if self.trade_count:
            self.high = max(self.high, t.price)
            self.low = min(self.low, t.price)
        self.close = t.price
        self.volume += t.volume
        self.value += t.price * t.volume
        if t.side == "BID":
            self.buy_volume += t.volume
        else:
            self.sell_volume += t.volume
        self.trade_count += 1
        if t.has_book:  # keep the last valid snapshot inside the minute
            self.bid_price, self.ask_price = t.bid_price, t.ask_price
            self.bid_size, self.ask_size = t.bid_size, t.ask_size
            self.total_bid, self.total_ask = t.total_bid, t.total_ask

    @property
    def iso(self) -> str:
        return minute_iso(self.minute)


def minute_iso(minute: int) -> str:
    return datetime.fromtimestamp(minute * 60, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def aggregate_minutes(trades: Iterable[Trade]) -> list[MinuteBar]:
    """Trades must be sorted by time."""
    bars: list[MinuteBar] = []
    current: MinuteBar | None = None
    for t in trades:
        m = int(t.ts // 60)
        if current is None or m != current.minute:
            if current is not None:
                bars.append(current)
            current = MinuteBar.start(t)
        else:
            current.add(t)
    if current is not None:
        bars.append(current)
    return bars


class MinuteAggregator:
    """Streaming version of aggregate_minutes used by the realtime pipeline."""

    def __init__(self) -> None:
        self.current: MinuteBar | None = None
        self.late_trades = 0

    def add(self, t: Trade) -> list[MinuteBar]:
        m = int(t.ts // 60)
        if self.current is None:
            self.current = MinuteBar.start(t)
            return []
        if m == self.current.minute:
            self.current.add(t)
            return []
        if m < self.current.minute:
            self.late_trades += 1  # arrived after its minute was closed
            return []
        done, self.current = self.current, MinuteBar.start(t)
        return [done]

    def flush_due(self, now_ts: float, grace_seconds: float = 3.0) -> list[MinuteBar]:
        """Close the current bar when its minute is over (even if no new trade came)."""
        if self.current is not None and now_ts >= (self.current.minute + 1) * 60 + grace_seconds:
            done, self.current = self.current, None
            return [done]
        return []


def fill_gaps(bars: list[MinuteBar], max_gap: int = MAX_FILL_GAP) -> tuple[list[MinuteBar], np.ndarray]:
    """Insert flat zero-volume bars for short gaps; return bars and segment ids.

    A gap longer than `max_gap` missing minutes (e.g. collector stopped) starts a
    new segment; windows are never built across segments.
    """
    out: list[MinuteBar] = []
    segments: list[int] = []
    seg = 0
    for bar in sorted(bars, key=lambda b: b.minute):
        if out:
            prev = out[-1]
            gap = bar.minute - prev.minute - 1
            if gap < 0:  # duplicate minute (should not happen after aggregation)
                continue
            if gap > max_gap:
                seg += 1
            else:
                for k in range(1, gap + 1):
                    out.append(MinuteBar(minute=prev.minute + k, open=prev.close, high=prev.close,
                                         low=prev.close, close=prev.close, bid_price=prev.bid_price,
                                         ask_price=prev.ask_price, bid_size=prev.bid_size,
                                         ask_size=prev.ask_size, total_bid=prev.total_bid,
                                         total_ask=prev.total_ask, filled=True))
                    segments.append(seg)
        out.append(bar)
        segments.append(seg)
    return out, np.asarray(segments, dtype=np.int64)


@dataclass
class FeatureFrame:
    minutes: np.ndarray        # (N,) epoch minutes
    segments: np.ndarray       # (N,)
    X: np.ndarray              # (N, F) features, NaN where not computable
    valid: np.ndarray          # (N,) all features finite
    close: np.ndarray
    volume: np.ndarray
    value: np.ndarray
    trade_count: np.ndarray
    filled: np.ndarray

    def __len__(self) -> int:
        return len(self.minutes)

    def row_features(self, i: int) -> dict:
        return {name: (None if not math.isfinite(v) else float(v)) for name, v in zip(FEATURES, self.X[i])}


def build_features(bars: list[MinuteBar], max_gap: int = MAX_FILL_GAP) -> FeatureFrame:
    bars, seg = fill_gaps(bars, max_gap)
    n = len(bars)
    col = lambda name: np.array([getattr(b, name) for b in bars], dtype=float)  # noqa: E731
    o, h, lo, c = col("open"), col("high"), col("low"), col("close")
    vol, val, cnt = col("volume"), col("value"), col("trade_count")
    buy, sell = col("buy_volume"), col("sell_volume")
    bp, ap, bs, as_ = col("bid_price"), col("ask_price"), col("bid_size"), col("ask_size")
    tb, ta = col("total_bid"), col("total_ask")

    # Forward-fill orderbook snapshot inside each segment (bars whose trades all
    # lacked a snapshot). Never fill backwards: that would leak future data.
    for arr in (bp, ap, bs, as_, tb, ta):
        for i in range(1, n):
            if np.isnan(arr[i]) and seg[i] == seg[i - 1]:
                arr[i] = arr[i - 1]

    same_prev = np.zeros(n, dtype=bool)
    if n > 1:
        same_prev[1:] = seg[1:] == seg[:-1]
    prev = lambda a: np.concatenate([[np.nan], a[:-1]]) if n else a  # noqa: E731

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(same_prev, np.log(c / prev(c)), np.nan)
        hl = np.log(h / lo)
        lv = np.log1p(vol * 1000.0)            # volume in mBTC
        lval = np.log1p(val / 1e6)             # value in million KRW
        d_lv = np.where(same_prev, lv - prev(lv), np.nan)
        d_lval = np.where(same_prev, lval - prev(lval), np.nan)
        lcnt = np.log1p(cnt)
        tot = buy + sell
        bsi = np.where(tot > 0, (buy - sell) / tot, 0.0)
        mid = (ap + bp) / 2.0
        spread = (ap - bp) / mid * 1e4
        l1 = (bs - as_) / (bs + as_)
        depth = (tb - ta) / (tb + ta)

    rv = np.full(n, np.nan)
    for i in range(n):
        j = i - RV_WINDOW + 1
        if j < 0 or seg[j] != seg[i]:
            continue
        w = log_ret[j:i + 1]
        if np.all(np.isfinite(w)):
            rv[i] = float(np.std(w))

    X = np.column_stack([log_ret, hl, rv, d_lv, lval, d_lval, lcnt, bsi, spread, l1, depth]) if n \
        else np.zeros((0, len(FEATURES)))
    X[~np.isfinite(X)] = np.nan
    valid = np.all(np.isfinite(X), axis=1) if n else np.zeros(0, dtype=bool)
    return FeatureFrame(minutes=np.array([b.minute for b in bars], dtype=np.int64), segments=seg, X=X,
                        valid=valid, close=c, volume=vol, value=val, trade_count=cnt,
                        filled=np.array([b.filled for b in bars], dtype=bool))


def window_end_indices(frame: FeatureFrame, window: int) -> np.ndarray:
    """Indices e such that rows e-window+1..e are valid and in one segment."""
    ends = []
    run = 0
    for i in range(len(frame)):
        if frame.valid[i] and (run == 0 or frame.segments[i] == frame.segments[i - 1]):
            run += 1
        elif frame.valid[i]:
            run = 1
        else:
            run = 0
        if run >= window:
            ends.append(i)
    return np.asarray(ends, dtype=np.int64)


def make_windows(X: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    if len(ends) == 0:
        return np.zeros((0, window, X.shape[1]), dtype=np.float32)
    idx = ends[:, None] + np.arange(-window + 1, 1)[None, :]
    return X[idx].astype(np.float32)


class Scaler:
    """z-score scaler fitted on training rows only; values are clipped."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 8.0) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.clip = float(clip)

    @classmethod
    def fit(cls, X: np.ndarray, clip: float = 8.0) -> "Scaler":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std < 1e-9, 1.0, std)  # constant feature -> keep raw deviations
        return cls(mean, std, clip)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -self.clip, self.clip)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(np.array(d["mean"]), np.array(d["std"]), d.get("clip", 8.0))
