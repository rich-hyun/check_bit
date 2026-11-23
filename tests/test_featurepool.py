"""Feature pool (checkbit_ai/featurepool). All market data here is SYNTHETIC."""
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from checkbit_ai import rawdb
from checkbit_ai.featurepool import params as P
from checkbit_ai.featurepool import store
from checkbit_ai.featurepool.definitions import FEATURE_NAMES, render_markdown
from checkbit_ai.featurepool.minute import coverage_ms
from checkbit_ai.featurepool.normalize import Normalizer, RollingWindow
from checkbit_ai.featurepool.pipeline import last_final_minute, process_minutes

M = "KRW-BTC"
BASE_MIN = 29_500_000          # an arbitrary epoch minute
MS = P.MINUTE_MS


def _ms(minute, sec=0.0):
    return int((BASE_MIN + minute) * MS + sec * 1000)


def _levels(mid, half_spread, tick=1000, n=15, size=1.0):
    bid1, ask1 = mid - half_spread, mid + half_spread
    return [(d, ask1 + d * tick, bid1 - d * tick, size, size) for d in range(n)]  # depth, ask, bid, asz, bsz


class RawBuilder:
    def __init__(self, path: Path):
        self.path = path
        rawdb.init_db(path)
        self.c = sqlite3.connect(path)
        self.seq = 0

    def session(self, a_ms, b_ms):
        self.c.execute("INSERT INTO collector_sessions (market, connected_at, disconnected_at, reason)"
                       " VALUES (?,?,?,'stop')", (M, a_ms, b_ms))

    def trade(self, ts, price, vol, side="BID"):
        self.seq += 1
        self.c.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)",
                       (M, self.seq, ts, price, vol, side, None, "REALTIME", ts))

    def snapshot(self, ts, levels, total=99.0):
        cur = self.c.execute("INSERT INTO orderbook_snapshots (market, timestamp, total_ask_size, total_bid_size,"
                             " level, stream_type, received_at) VALUES (?,?,?,?,0,'REALTIME',?)",
                             (M, ts, total, total, ts))
        sid = cur.lastrowid
        self.c.executemany("INSERT INTO orderbook_units VALUES (?,?,?,?,?,?)",
                           [(sid, d, a, b, asz, bsz) for d, a, b, asz, bsz in levels])

    def done(self):
        self.c.commit()
        self.c.close()


def synthetic(path: Path, minutes: int, gaps=(), seed=0, trades_per_min=40):
    """Random-walk trades with bid-ask bounce, 1 snapshot per second, full coverage except `gaps`."""
    rng = np.random.default_rng(seed)
    rb = RawBuilder(path)
    edges = [0] + [x for g in sorted(gaps) for x in (g, g + 1)] + [minutes]
    for a, b in zip(edges[::2], edges[1::2]):           # sessions around the gap minutes
        if b > a:
            rb.session(_ms(a), _ms(b))
    mid = 100_000_000.0
    for m in range(minutes):
        for s in range(0, 60):
            rb.snapshot(_ms(m, s + 0.5), _levels(mid, 2000 + 1000 * (s % 3)))
        for k in range(trades_per_min):
            mid *= math.exp(rng.normal(0, 2e-5))
            side = "BID" if rng.random() < 0.5 else "ASK"
            price = round(mid / 1000) * 1000 + (1000 if side == "BID" else -1000)
            rb.trade(_ms(m, (k + 0.5) * 60 / trades_per_min), price, round(float(rng.uniform(0.001, 0.05)), 6), side)
    rb.done()
    return path


def _values(feat_path, minute):
    c = sqlite3.connect(feat_path)
    try:
        return {f: v for f, v in c.execute("SELECT feature, value FROM feature_values WHERE market=? AND minute=?",
                                           (M, BASE_MIN + minute))}
    finally:
        c.close()


def _run(raw_path, feat_path, m0, m1, now=None):
    raw = sqlite3.connect(raw_path)
    feat = store.connect(feat_path)
    try:
        now = now if now is not None else _ms(m1 + 2)
        return process_minutes(raw, feat, M, BASE_MIN + m0, BASE_MIN + m1, source="replay", now=now)
    finally:
        raw.close()
        feat.close()


# ------------------------------------------------------------- hand-checked minute
def test_single_minute_values(tmp_path):
    rb = RawBuilder(tmp_path / "raw.sqlite3")
    rb.session(_ms(0), _ms(3))
    # minute 0: one trade so minute 1 has a previous close
    rb.trade(_ms(0, 10), 100_000_000, 0.01, "BID")
    # minute 1
    rb.trade(_ms(1, 1), 100_000_000, 0.2, "BID")     # 20,000,000 KRW -> large; 0.2 is a 0.001 multiple
    rb.trade(_ms(1, 2), 100_100_000, 0.0123456, "ASK")
    rb.trade(_ms(1, 3), 100_050_000, 0.1, "BID")      # 10,005,000 KRW -> large (>= 10M)
    lv_wide = _levels(100_000_000, 2000)              # spread 4000 > tick 1000
    lv_tight = [(d, 100_000_500 + d * 1000, 99_999_500 - d * 1000, 1.0, 3.0 if d < 5 else 1.0) for d in range(15)]
    rb.snapshot(_ms(1, 5), lv_wide)
    rb.snapshot(_ms(1, 6), lv_tight)                  # spread 1000 == tick -> not wide
    rb.done()
    _run(tmp_path / "raw.sqlite3", tmp_path / "f.sqlite3", 0, 2)
    v = _values(tmp_path / "f.sqlite3", 1)
    vol = 0.2 + 0.0123456 + 0.1
    krw = 100_000_000 * 0.2 + 100_100_000 * 0.0123456 + 100_050_000 * 0.1
    assert v["volume"] == pytest.approx(vol) and v["volume_krw"] == pytest.approx(krw)
    assert v["trade_count"] == 3
    assert v["buy_ratio"] == pytest.approx(0.3 / vol)
    assert v["net_taker_volume"] == pytest.approx(0.3 - 0.0123456)
    assert v["average_trade_size"] == pytest.approx(vol / 3)
    assert v["large_trade_ratio"] == pytest.approx(0.3 / vol)
    assert v["round_number_ratio_btc"] == pytest.approx(2 / 3)
    assert v["price_return_1m"] == pytest.approx(math.log(100_050_000 / 100_000_000))
    assert v["price_range_1m"] == pytest.approx(math.log(100_100_000 / 100_000_000))
    assert v["amihud"] == pytest.approx(abs(math.log(100_050_000 / 1e8)) / (krw / 1e8))
    s_wide, s_tight = 4000 / 1e8 * 1e4, 1000 / 1e8 * 1e4
    assert v["spread_mean_bp"] == pytest.approx((s_wide + s_tight) / 2)
    assert v["spread_max_bp"] == pytest.approx(s_wide)
    assert v["spread_wide_ratio"] == pytest.approx(0.5)
    assert v["depth_total_15"] == pytest.approx((30 + (5 * 4 + 10 * 2)) / 2)  # summed levels, not total_*_size=99
    assert v["order_imbalance_5"] == pytest.approx((0 + (15 - 5) / 20) / 2)
    assert v["volume_change_rate"] == pytest.approx(math.log(vol + 0.001) - math.log(0.01 + 0.001))


# ------------------------------------------------------------- gaps and NaN rules
def test_coverage_union():
    assert coverage_ms([(0, 30), (20, 50), (55, 70)], 0, 60) == 55


def test_gap_tolerance_and_nan_propagation(tmp_path):
    raw = tmp_path / "raw.sqlite3"
    synthetic(raw, minutes=60, gaps=(30,))
    # add a 2 s hole inside minute 10 (tolerated) by splitting the first session
    c = sqlite3.connect(raw)
    c.execute("UPDATE collector_sessions SET disconnected_at=? WHERE connected_at=?", (_ms(10, 20), _ms(0)))
    c.execute("INSERT INTO collector_sessions (market, connected_at, disconnected_at) VALUES (?,?,?)",
              (M, _ms(10, 22), _ms(30)))
    c.commit()
    c.close()
    feat = tmp_path / "f.sqlite3"
    _run(raw, feat, 0, 59)
    fc = sqlite3.connect(feat)
    gaps = {m - BASE_MIN for (m,) in fc.execute("SELECT minute FROM feature_minutes WHERE is_gap=1")}
    fc.close()
    assert gaps == {30}
    assert all(v is None for v in _values(feat, 30).values())            # gap: every feature NaN
    assert _values(feat, 31)["volume_change_rate"] is None                 # needs t-1
    assert _values(feat, 31)["price_return_1m"] is None
    assert _values(feat, 45)["volatility"] is None                         # r_31 = ln(c31/c30) undefined
    assert _values(feat, 46)["volatility"] is not None                     # r_32..r_46
    assert _values(feat, 59)["roll_measure"] is None                      # window 30..59 contains the gap
    assert _values(feat, 29)["roll_measure"] is not None                   # window 0..29 has >= 200 changes


def test_return_needs_both_endpoints(tmp_path):
    raw = tmp_path / "raw.sqlite3"
    synthetic(raw, minutes=40, gaps=(30,))
    _run(raw, tmp_path / "f.sqlite3", 0, 39)
    assert _values(tmp_path / "f.sqlite3", 35)["price_return_5m"] is None  # close_30 undefined
    assert _values(tmp_path / "f.sqlite3", 34)["price_return_5m"] is not None


def test_no_trade_minute_takes_previous_close_but_not_across_gap(tmp_path):
    rb = RawBuilder(tmp_path / "raw.sqlite3")
    rb.session(_ms(0), _ms(3))
    rb.session(_ms(4), _ms(6))
    rb.trade(_ms(0, 1), 100_000_000, 0.01)
    rb.trade(_ms(2, 1), 101_000_000, 0.01)            # minute 1: no trades
    rb.trade(_ms(5, 1), 102_000_000, 0.01)            # minute 3: gap, minute 4: no trades
    rb.done()
    _run(tmp_path / "raw.sqlite3", tmp_path / "f.sqlite3", 0, 5)
    v1, v2, v4 = (_values(tmp_path / "f.sqlite3", m) for m in (1, 2, 4))
    assert v1["price_return_1m"] == pytest.approx(0.0) and v1["volume"] == 0 and v1["buy_ratio"] is None
    assert v2["price_return_1m"] == pytest.approx(math.log(1.01))
    assert v4["price_return_1m"] is None                                   # close_4 has no source (gap at 3)


# ------------------------------------------------------------- formulas
def test_bipower_matches_lee_mykland_definition(tmp_path):
    raw = tmp_path / "raw.sqlite3"
    n = P.BIPOWER_K + 5
    synthetic(raw, minutes=n, trades_per_min=3)
    feat = tmp_path / "f.sqlite3"
    _run(raw, feat, 0, n - 1)
    c = sqlite3.connect(feat)
    r = {m - BASE_MIN: v for m, v in c.execute(
        "SELECT minute, value FROM feature_values WHERE feature='price_return_1m'")}
    c.close()
    t, K = n - 1, P.BIPOWER_K
    expected = sum(abs(r[j]) * abs(r[j - 1]) for j in range(t - K + 2, t)) / (K - 2)
    assert _values(feat, t)["bipower_var"] == pytest.approx(expected)
    assert _values(feat, K - 1)["bipower_var"] is None                     # needs r_0 which is undefined


def test_roll_recovers_bid_ask_bounce(tmp_path):
    rng = np.random.default_rng(1)
    rb = RawBuilder(tmp_path / "raw.sqlite3")
    rb.session(_ms(0), _ms(31))
    spread = 2000.0                                    # trades bounce +-1000 around a fixed mid
    for m in range(30):
        for k in range(20):
            side = "BID" if rng.random() < 0.5 else "ASK"
            rb.trade(_ms(m, k * 3 + 1), 1e8 + (spread / 2 if side == "BID" else -spread / 2), 0.01, side)
    rb.done()
    _run(tmp_path / "raw.sqlite3", tmp_path / "f.sqlite3", 0, 29)
    roll = _values(tmp_path / "f.sqlite3", 29)["roll_measure"]
    assert roll == pytest.approx(spread / 1e8 * 1e4, rel=0.15)


# ------------------------------------------------------------- normalization
def test_rolling_window_excludes_current_and_applies_floor():
    s = np.array([1.0] * 40 + [100.0])
    ref = RollingWindow("t", 60, 30)
    assert ref.baseline(s, 40) == (1.0, 0.0)                               # spike itself is not in its baseline
    zs, zl, z = Normalizer(short=ref, long=RollingWindow("l", 1440, 720)).z(s, 40, floor=0.5)
    assert zs == pytest.approx((100 - 1) / 0.5) and math.isnan(zl) and z == zs  # long NaN -> short


def test_long_z_used_when_available():
    rng = np.random.default_rng(2)
    s = rng.normal(size=800)
    zs, zl, z = Normalizer().z(s, 799, floor=1e-9)
    assert math.isfinite(zl) and z == zl and zs != zl


# ------------------------------------------------------------- realtime == replay
def test_realtime_minute_by_minute_equals_replay(tmp_path):
    raw = tmp_path / "raw.sqlite3"
    synthetic(raw, minutes=100, gaps=(50,), trades_per_min=12)
    rt, rp = tmp_path / "rt.sqlite3", tmp_path / "rp.sqlite3"
    for m in range(100):                                # as the service would: each minute after it closes
        assert _run(raw, rt, m, m, now=_ms(m + 1, 3.0)) == 1
    _run(raw, rp, 0, 99)
    q = "SELECT minute, feature, value, z_short, z_long, z FROM feature_values ORDER BY minute, feature"
    a, b = sqlite3.connect(rt).execute(q).fetchall(), sqlite3.connect(rp).execute(q).fetchall()
    assert len(a) == 100 * len(FEATURE_NAMES) and a == b


def test_unfinished_minute_is_not_computed(tmp_path):
    raw = tmp_path / "raw.sqlite3"
    synthetic(raw, minutes=3)
    assert last_final_minute(_ms(2, 2.9)) == BASE_MIN + 0                  # minute 1 needs t+60s+3s
    assert _run(raw, tmp_path / "f.sqlite3", 0, 2, now=_ms(2, 3.0)) == 2


# ------------------------------------------------------------- docs
def test_features_doc_is_in_sync():
    doc = Path(__file__).resolve().parent.parent / "docs" / "FEATURES.md"
    assert doc.read_text(encoding="utf-8") == render_markdown(), \
        "run: python -m checkbit_ai.featurepool.docs"
