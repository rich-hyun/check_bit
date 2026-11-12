import math

import numpy as np

from checkbit_ai.features import (FEATURES, CleanStats, MinuteAggregator, Scaler, Trade, aggregate_minutes,
                                  build_features, load_trades, parse_trade_row, window_end_indices)

BASE = 1_790_000_040  # aligned to a minute boundary (divisible by 60)


def row(**kw):
    r = {"timestamp": "2026-09-08 13:41:53.911553", "market": "KRW-BTC", "trade_price": "100000000",
         "trade_volume": "0.01", "ask_bid": "BID", "bid_price": "99999000", "ask_price": "100000000",
         "bid_size": "0.5", "ask_size": "0.4", "total_bid_size": "5", "total_ask_size": "6",
         "trade_timestamp": "", "sequential_id": ""}
    r.update({k: str(v) for k, v in kw.items()})
    return r


def trade(ts, price=100_000_000.0, vol=0.01, side="BID", book=True):
    t = Trade(ts=ts, price=price, volume=vol, side=side)
    if book:
        t.bid_price, t.ask_price, t.bid_size, t.ask_size, t.total_bid, t.total_ask = \
            price - 1000, price, 0.5, 0.4, 5.0, 6.0
    return t


def test_parse_validation_and_missing_orderbook():
    st = CleanStats()
    assert parse_trade_row(row(trade_price=0), "KRW-BTC", st) is None
    assert parse_trade_row(row(trade_volume=-1), "KRW-BTC", st) is None
    assert parse_trade_row(row(ask_bid="X"), "KRW-BTC", st) is None
    assert parse_trade_row(row(timestamp="garbage"), "KRW-BTC", st) is None
    assert parse_trade_row(row(market="KRW-ETH"), "KRW-BTC", st) is None
    t = parse_trade_row(row(bid_price=0, ask_price=0, bid_size=0, ask_size=0, total_bid_size=0, total_ask_size=0),
                        "KRW-BTC", st)
    assert t is not None and not t.has_book  # zeros = no snapshot yet -> missing, not 0
    assert st.invalid == 4 and st.other_market == 1 and st.missing_book == 1


def test_trade_timestamp_preferred_over_receive_time():
    t = parse_trade_row(row(trade_timestamp=1794442909868), "KRW-BTC", CleanStats())
    assert t.ts == 1794442909.868


def test_dedup_by_sequential_id_and_exact_rows(tmp_path):
    import csv
    from checkbit_ai.features import CSV_HEADER
    p = tmp_path / "a.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        r1 = row(trade_timestamp=BASE * 1000, sequential_id="1")
        w.writerow([r1[k] for k in CSV_HEADER])
        w.writerow([r1[k] for k in CSV_HEADER])  # duplicate id
        r2 = row(trade_timestamp=(BASE + 1) * 1000, sequential_id="2")
        w.writerow([r2[k] for k in CSV_HEADER])
        legacy = row()
        w.writerow([legacy[k] for k in CSV_HEADER])
        w.writerow([legacy[k] for k in CSV_HEADER])  # exact duplicate legacy row
        w.writerow(["broken"] * 15)                  # too many columns
    trades, st = load_trades([p])
    assert len(trades) == 3 and st.duplicates == 2 and st.reasons.get("column_count") == 1


def test_minute_aggregation_and_streaming_match():
    ts = [BASE + 1, BASE + 20, BASE + 59, BASE + 61, BASE + 130]
    trades = [trade(t, price=100_000_000 + i * 1000, vol=0.1 * (i + 1), side="BID" if i % 2 == 0 else "ASK")
              for i, t in enumerate(ts)]
    bars = aggregate_minutes(trades)
    assert [b.trade_count for b in bars] == [3, 1, 1]
    b0 = bars[0]
    assert b0.open == 100_000_000 and b0.close == 100_002_000 and b0.high == 100_002_000
    assert math.isclose(b0.buy_volume, 0.1 + 0.3) and math.isclose(b0.sell_volume, 0.2)
    agg = MinuteAggregator()
    done = []
    for t in trades:
        done += agg.add(t)
    done += agg.flush_due(BASE + 300)
    assert [(b.minute, b.trade_count, b.close) for b in done] == [(b.minute, b.trade_count, b.close) for b in bars]


def test_gap_fill_and_segments():
    trades = [trade(BASE + m * 60 + 5) for m in (0, 1, 4, 30, 31)]  # 2-minute gap filled, 25-minute gap splits
    f = build_features(aggregate_minutes(trades))
    assert len(f) == 7 and f.filled.sum() == 2
    assert f.segments.tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert f.volume[2] == 0 and f.close[2] == f.close[1]


def _series(n=80, seed=0):
    rng = np.random.default_rng(seed)
    trades, p = [], 100_000_000.0
    for m in range(n):
        for k in range(5):
            p *= math.exp(rng.normal(0, 1e-4))
            trades.append(trade(BASE + m * 60 + k * 10, price=round(p), vol=float(rng.uniform(0.001, 0.1)),
                                side="BID" if rng.random() < .5 else "ASK"))
    return trades


def test_features_are_causal():
    trades = _series()
    full = build_features(aggregate_minutes(trades))
    cut_minute = int(trades[0].ts // 60) + 50
    part = build_features(aggregate_minutes([t for t in trades if t.ts // 60 <= cut_minute]))
    n = len(part)
    np.testing.assert_allclose(full.X[:n], part.X, equal_nan=True)  # future data does not change the past


def test_feature_values_and_windows():
    trades = _series()
    f = build_features(aggregate_minutes(trades))
    assert f.X.shape[1] == len(FEATURES)
    assert not f.valid[:10].any() and f.valid[10:].all()  # rv_10 needs 10 returns
    i = 20
    assert math.isclose(f.X[i, 0], math.log(f.close[i] / f.close[i - 1]))
    spread = f.X[i, FEATURES.index("spread_bps")]
    assert spread > 0
    ends = window_end_indices(f, 20)
    assert ends[0] == 29 and len(ends) == len(f) - 29


def test_scaler_clip_and_constant_feature():
    X = np.array([[1.0, 5.0], [3.0, 5.0]])
    s = Scaler.fit(X)
    assert s.std[1] == 1.0
    out = s.transform(np.array([[1000.0, 5.0]]))
    assert out[0, 0] == s.clip and out[0, 1] == 0.0
    assert Scaler.from_dict(s.to_dict()).transform(X).tolist() == s.transform(X).tolist()
