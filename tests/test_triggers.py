"""Triggers (checkbit_ai/triggers). All market data here is SYNTHETIC."""
import json
import math
import sqlite3

import numpy as np
import pytest

from checkbit_ai.featurepool import store as fstore
from checkbit_ai.triggers import candidates, jump, lstm, robust_z
from checkbit_ai.triggers import params as P
from checkbit_ai.triggers import store as tstore
from checkbit_ai.triggers.pipeline import process_triggers, trigger_range
from test_featurepool import BASE_MIN, M, RawBuilder, _levels, _ms, _run

# ---------------------------------------------------------------- helpers
def feat_db(path):
    conn = fstore.connect(path)
    tstore.ensure(conn)
    return conn


def put(conn, minute, values=None, *, is_gap=False, n_trades=100, low=False):
    """One feature minute; values: feature -> z (value = z for simplicity)."""
    conn.execute("INSERT OR REPLACE INTO feature_minutes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (M, minute, 60000, int(is_gap), n_trades, 60, int(low), 0, "test", "replay", 0))
    for f, z in (values or {}).items():
        conn.execute("INSERT OR REPLACE INTO feature_values VALUES (?,?,?,?,?,?,?)", (M, minute, f, z, z, z, z))


def analyses(path, rows):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE analyses (id INTEGER PRIMARY KEY, market TEXT, minute INTEGER, score REAL,"
              " threshold REAL, is_anomaly INTEGER, model_version TEXT)")
    c.executemany("INSERT INTO analyses (market, minute, score, threshold, is_anomaly, model_version)"
                  " VALUES (?,?,?,?,?,'v1')", [(M, m, s, t, int(s > t)) for m, s, t in rows])
    c.commit()
    c.close()


def cands(conn):
    return conn.execute("SELECT start_minute, end_minute, peak_minute, status, tier, detectors, types,"
                        " hit_minutes, lstm_minutes, agent_status FROM candidates ORDER BY start_minute").fetchall()


# ---------------------------------------------------------------- jump
def test_gumbel_threshold_is_about_4_6_sigma():
    t_l = jump.CN + jump.BETA * jump.SN          # |L| at the decision boundary
    assert 4.55 < jump.C * t_l < 4.7             # in c·L (≈ N(0,1)) units
    bv = 1e-8
    below = jump.evaluate(0.999 * t_l * math.sqrt(bv), bv)
    above = jump.evaluate(1.001 * t_l * math.sqrt(bv), bv)
    assert not below["is_anomaly"] and above["is_anomaly"]
    assert jump.evaluate(None, bv) is None and jump.evaluate(0.001, 0.0) is None


def test_jump_false_alarms_under_no_jumps():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1e-4, 200_000)             # ~139 days of 1-minute returns, no jumps
    K = 120
    a = np.abs(r)
    prod = a[1:] * a[:-1]
    csum = np.concatenate([[0], np.cumsum(prod)])
    hits = 0
    for t in range(K, len(r)):                   # products j = t-K+2 .. t-1  -> prod index j-1
        bv = (csum[t - 1] - csum[t - K + 1]) / (K - 2)
        hits += jump.evaluate(r[t], bv)["is_anomaly"]
    assert hits <= 10                            # expectation ~ alpha * days ~ 1.4


def test_bh_reject():
    mask = jump.bh_reject(np.array([0.001, 0.02, 0.03, 0.5]), q=0.05)
    assert mask.tolist() == [True, True, True, False]
    assert jump.bh_reject(np.array([0.2, 0.5]), q=0.05).tolist() == [False, False]


def test_jump_detected_end_to_end_from_raw(tmp_path):
    rng = np.random.default_rng(3)
    raw = tmp_path / "raw.sqlite3"
    rb = RawBuilder(raw)
    n = 200
    rb.session(_ms(0), _ms(n))
    mid = 1e8
    for m in range(n):
        rb.snapshot(_ms(m, 30), _levels(mid, 2000))
        if m == 180:
            mid *= math.exp(0.004)               # ~35 sigma one-minute jump
        for k in range(30):
            mid *= math.exp(rng.normal(0, 2e-5))
            rb.trade(_ms(m, k * 2 + 0.5), round(mid), 0.01, "BID" if k % 2 else "ASK")
    rb.done()
    feat = tmp_path / "f.sqlite3"
    _run(raw, feat, 0, n - 1)
    conn = feat_db(feat)
    process_triggers(conn, None, M, BASE_MIN, BASE_MIN + n - 1)
    hits = [m - BASE_MIN for (m,) in conn.execute(
        "SELECT minute FROM detector_hits WHERE detector='jump' AND is_anomaly=1")]
    tested = conn.execute("SELECT COUNT(*) FROM detector_hits WHERE detector='jump'").fetchone()[0]
    assert hits == [180] and tested > 50
    ev = json.loads(conn.execute("SELECT evidence FROM detector_hits WHERE detector='jump' AND minute=?",
                                 (BASE_MIN + 180,)).fetchone()[0])[0]
    assert ev["direction"] == "up" and ev["cL"] > 10


# ---------------------------------------------------------------- robust z
def test_robust_z_directions_and_exclusions():
    v = lambda z: (z, z)  # noqa: E731
    out = robust_z.evaluate({"volume": v(5.0), "depth_total_15": v(5.0), "price_return_1m": v(9.0)}, 100, False)
    assert out["is_anomaly"] and [h["feature"] for h in out["evidence"]] == ["volume"]   # depth is down-only
    assert out["score"] == pytest.approx(5 / 4)
    assert robust_z.evaluate({"depth_total_15": v(-5.0)}, 100, False)["is_anomaly"]
    assert not robust_z.evaluate({"buy_ratio": v(-6.0), "volume": v(1.0)}, 10, False)["is_anomaly"]  # few trades
    assert robust_z.evaluate({"buy_ratio": v(-6.0)}, 10, False) is None                  # nothing judgeable
    assert not robust_z.evaluate({"spread_max_bp": v(8.0), "volume": v(0.0)}, 100, True)["is_anomaly"]
    assert robust_z.evaluate({"price_return_15m": v(-4.5)}, 100, False)["evidence"][0]["type_hint"] == "가격 급변"


# ---------------------------------------------------------------- LSTM wrapper
def test_lstm_wrapper(tmp_path):
    assert lstm.load(tmp_path / "missing.sqlite3", M, 0, 10) == {}
    empty = tmp_path / "empty.sqlite3"
    sqlite3.connect(empty).close()
    assert lstm.load(empty, M, 0, 10) == {}                                               # no table yet
    db = tmp_path / "a.sqlite3"
    analyses(db, [(5, 3.0, 2.5), (6, 1.0, 2.5), (7, 9.0, 2.5)])
    rows = lstm.load(db, M, 0, 10)
    assert lstm.evaluate(rows[5], False)["is_anomaly"] and not lstm.evaluate(rows[6], False)["is_anomaly"]
    assert lstm.evaluate(rows[7], True) is None                                           # gap minute ignored


# ---------------------------------------------------------------- candidates
def _feed(conn, hits: dict[int, float], n: int, gaps=(), lstm_db=None):
    """hits: minute -> volume z. Every other minute has volume z = 0."""
    for m in range(n):
        put(conn, m, {"volume": hits.get(m, 0.0)}, is_gap=m in gaps)
    conn.commit()
    for m in range(n):                                       # minute by minute, like realtime
        process_triggers(conn, lstm_db, M, m, m)


def test_merge_gap_close_timing_and_length_cap(tmp_path):
    conn = feat_db(tmp_path / "f.sqlite3")
    hits = {10: 5, 12: 5, 15: 5, 30: 5, 34: 5, **{m: 5 for m in range(40, 106)}}
    for m in range(18):
        put(conn, m, {"volume": hits.get(m, 0.0)})
        process_triggers(conn, None, M, m, m)
    assert cands(conn)[0][3] == "open"                       # 16, 17 are hit-free; still open
    put(conn, 18, {"volume": 0.0})
    process_triggers(conn, None, M, 18, 18)
    assert cands(conn)[0][:4] == (10, 15, 10, "closed")      # 3 hit-free minutes -> closed
    for m in range(19, 110):
        put(conn, m, {"volume": hits.get(m, 0.0)})
        process_triggers(conn, None, M, m, m)
    spans = [(a, b) for a, b, *_ in cands(conn)]
    assert spans == [(10, 15), (30, 30), (34, 34), (40, 99), (100, 105)]   # 30/34: 3 minutes apart; 60-minute cap


def test_tiers_and_types(tmp_path):
    db = tmp_path / "a.sqlite3"
    analyses(db, [(50, 3.0, 2.5)])                            # LSTM hit inside the third candidate
    conn = feat_db(tmp_path / "f.sqlite3")
    _feed(conn, {10: 5.0, 30: 6.5, 50: 4.4}, 60, lstm_db=db)
    rows = cands(conn)
    assert [(r[0], r[4]) for r in rows] == [(10, "ambiguous"), (30, "strong"), (50, "strong")]
    assert json.loads(rows[0][6]) == ["거래량 급증"]
    assert set(json.loads(rows[2][5])) == {"robust_z", "lstm"} and P.LSTM_TYPE in json.loads(rows[2][6])
    assert rows[2][8] == 1 and rows[0][8] == 0                # lstm_minutes


def test_trigger_range_lags_features(tmp_path):
    conn = feat_db(tmp_path / "f.sqlite3")
    for m in range(5):
        put(conn, m)
    assert trigger_range(conn, M) == (0, 4 - P.LSTM_LAG_MINUTES)
    process_triggers(conn, None, M, 0, 3)
    assert trigger_range(conn, M) is None
    put(conn, 5)
    assert trigger_range(conn, M) == (4, 4)


def test_realtime_equals_replay_rebuild_and_keeps_agent_status(tmp_path):
    rng = np.random.default_rng(5)
    db = tmp_path / "a.sqlite3"
    n = 300
    analyses(db, [(m, float(rng.uniform(0, 4)), 2.5) for m in range(n) if rng.random() < 0.05])
    hits = {int(m): float(rng.uniform(4, 7)) for m in rng.choice(n, 25, replace=False)}
    rt = feat_db(tmp_path / "rt.sqlite3")
    _feed(rt, hits, n, gaps=(100, 101), lstm_db=db)
    rp = feat_db(tmp_path / "rp.sqlite3")
    for m in range(n):
        put(rp, m, {"volume": hits.get(m, 0.0)}, is_gap=m in (100, 101))
    rp.commit()
    process_triggers(rp, db, M, 0, n - 1, update_candidates=False)
    with rp:
        candidates.rebuild(rp, M, n - 1)
    q = "SELECT minute, detector, score, threshold, is_anomaly, evidence FROM detector_hits ORDER BY minute, detector"
    assert rt.execute(q).fetchall() == rp.execute(q).fetchall()
    assert cands(rt) == cands(rp) and len(cands(rt)) > 5
    # rebuilding keeps the agent queue state of unchanged intervals
    first = cands(rp)[0]
    rp.execute("UPDATE candidates SET agent_status='done' WHERE start_minute=?", (first[0],))
    with rp:
        candidates.rebuild(rp, M, n - 1)
    assert cands(rp)[0][-1] == "done"
