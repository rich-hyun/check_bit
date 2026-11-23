"""Raw SQLite store (checkbit_ai/rawdb.py) wired into the collector. Uses a local
fake Upbit server; all market data here is SYNTHETIC."""
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest
from websockets.sync.server import serve

from checkbit_ai.rawdb import CollectorLockedError, RawDbWriter
from checkbit_ai.rawstore import RawCsvStore
from collector import UpbitCollector
from conftest import retry_winsock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _book(ts, bid=99_999_000, ask=100_000_000, levels=15):
    return {"type": "orderbook", "code": "KRW-BTC", "timestamp": ts, "total_bid_size": 3.0, "total_ask_size": 4.0,
            "stream_type": "REALTIME", "level": 0,
            "orderbook_units": [{"bid_price": bid - i * 1000, "ask_price": ask + i * 1000,
                                 "bid_size": 0.1 + i, "ask_size": 0.2 + i} for i in range(levels)]}


def _trade(sid, ts):
    return {"type": "trade", "code": "KRW-BTC", "trade_price": 100_000_000, "trade_volume": 0.01,
            "ask_bid": "BID", "trade_timestamp": ts, "sequential_id": sid, "stream_type": "REALTIME"}


class FakeUpbit:
    """Per connection: a few orderbook updates (two inside the sampling interval,
    one duplicate timestamp), 5 trades (one delivered twice), plus trade 1 of the
    previous connection again (Upbit resends a SNAPSHOT trade on reconnect)."""

    def __init__(self, drop_after=None):
        self.connections = 0
        self.open_now = 0
        self.drop_after = drop_after
        self.subscriptions = []
        self.server = retry_winsock(lambda: serve(self.handler, "127.0.0.1", 0))
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(self, ws):
        self.connections += 1
        self.open_now += 1
        conn = self.connections
        base = 1_790_000_000_000 + conn * 60_000
        try:
            self.subscriptions.append(json.loads(ws.recv()))
            for ts in (base, base + 300, base + 300, base + 1000, base + 2500):
                ws.send(json.dumps(_book(ts)).encode())
            if conn > 1:
                ws.send(json.dumps(_trade((conn - 1) * 100 + 1, base - 60_000)).encode())
            for i in range(1, 6):
                ws.send(json.dumps(_trade(conn * 100 + i, base + i)).encode())
                if i == 2:
                    ws.send(json.dumps(_trade(conn * 100 + i, base + i)).encode())
            if self.drop_after is not None and conn <= self.drop_after:
                time.sleep(0.3)
                return  # server drops the connection
            while True:
                ws.recv()
        except Exception:
            pass
        finally:
            self.open_now -= 1

    def close(self):
        self.server.shutdown()


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _q(db, sql, *args):
    c = sqlite3.connect(db)
    try:
        return c.execute(sql, args).fetchall()
    finally:
        c.close()


def _make(tmp_path, url, interval=1000):
    db = tmp_path / "raw.sqlite3"
    rawdb = RawDbWriter(db, ["KRW-BTC"], interval)
    c = UpbitCollector(RawCsvStore(tmp_path / "raw.csv", tmp_path / "bk"), url=url, rawdb=rawdb)
    return c, rawdb, db


def test_trades_orderbook_and_sessions(tmp_path):
    fake = FakeUpbit()
    c, rawdb, db = _make(tmp_path, fake.url)
    try:
        assert c.start() is True
        assert _wait(lambda: c.store.rows == 5)
        assert fake.subscriptions[0][2]["codes"] == ["KRW-BTC.15"]  # full depth requested
        assert c.stop() is True
    finally:
        c.stop()
        fake.close()
    st = rawdb.status()
    assert st["trades"] == 5 and st["duplicate_trades"] == 1
    # base, base+1000, base+2500 kept; base+300 (x2) sampled out by the 1000 ms interval
    assert st["orderbook_snapshots"] == 3 and st["sampled_out"] == 2
    assert _q(db, "SELECT COUNT(*), COUNT(DISTINCT sequential_id) FROM trades") == [(5, 5)]
    assert _q(db, "SELECT COUNT(*) FROM orderbook_units") == [(45,)]
    assert _q(db, "SELECT MIN(n), MAX(n) FROM (SELECT COUNT(*) n FROM orderbook_units GROUP BY snapshot_id)") \
        == [(15, 15)]
    assert _q(db, "SELECT depth, bid_price, ask_price FROM orderbook_units WHERE depth IN (0, 14) "
                  "AND snapshot_id = 1 ORDER BY depth") == [(0, 99_999_000, 100_000_000), (14, 99_985_000, 100_014_000)]
    sessions = _q(db, "SELECT market, disconnected_at IS NOT NULL, reason FROM collector_sessions")
    assert sessions == [("KRW-BTC", 1, "stop")]


def test_dedup_survives_restart_and_reconnect_sessions(tmp_path):
    fake = FakeUpbit(drop_after=1)
    c, rawdb, db = _make(tmp_path, fake.url)
    try:
        c.start()
        assert _wait(lambda: fake.connections >= 2 and rawdb.status()["trades"] == 10, timeout=15)
        c.stop()
        # a fresh writer on the same DB (process restart) still rejects known ids
        c2, rawdb2, _ = _make(tmp_path, fake.url)
        c2.start()
        assert _wait(lambda: rawdb2.status()["duplicate_trades"] >= 2, timeout=10)  # resent SNAPSHOT + in-batch dup
        c2.stop()
    finally:
        c.stop()
        fake.close()
    assert _q(db, "SELECT COUNT(*) = COUNT(DISTINCT sequential_id) FROM trades") == [(1,)]
    reasons = [r for (r,) in _q(db, "SELECT reason FROM collector_sessions ORDER BY id")]
    assert reasons[0] == "closed" or reasons[0].startswith("error:")  # server dropped connection 1
    assert reasons[-1] == "stop" and len(reasons) == 3
    assert _q(db, "SELECT COUNT(*) FROM collector_sessions WHERE disconnected_at IS NULL") == [(0,)]


def test_duplicate_snapshot_skips_units():
    from checkbit_ai.rawdb import connect, init_db, insert_orderbook
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "x.sqlite3"
        init_db(db)
        conn = connect(db)
        snap = ("KRW-BTC", 1000, 1.0, 1.0, 0, "REALTIME", 1)
        assert insert_orderbook(conn, snap, [(0, 2.0, 1.0, 1.0, 1.0)]) is True
        conn.execute("INSERT INTO orderbook_snapshots (market, timestamp, total_ask_size, total_bid_size, received_at)"
                     " VALUES ('KRW-BTC', 2000, 1, 1, 1)")  # moves lastrowid on
        assert insert_orderbook(conn, snap, [(0, 9.0, 9.0, 9.0, 9.0)]) is False
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM orderbook_units").fetchone()[0] == 1
        conn.close()


def test_incomplete_messages_are_skipped(tmp_path):
    rawdb = RawDbWriter(tmp_path / "raw.sqlite3", ["KRW-BTC"])
    rawdb.start()
    rawdb.submit({"type": "orderbook", "code": "KRW-BTC", "total_bid_size": 1, "total_ask_size": 1,
                  "orderbook_units": [{"bid_price": 1, "ask_price": 2, "bid_size": 1, "ask_size": 1}]})  # no timestamp
    rawdb.submit({"type": "trade", "code": "KRW-BTC", "trade_price": 1})
    rawdb.stop()
    assert rawdb.status()["skipped_invalid"] == 2 and rawdb.status()["write_errors"] == 0


def test_cross_process_lock_and_dangling_sessions(tmp_path):
    db = tmp_path / "raw.sqlite3"
    code = f"""
import sys, time; sys.path.insert(0, {ROOT!r})
from checkbit_ai.rawdb import RawDbWriter
w = RawDbWriter({str(db)!r}, ['KRW-BTC'])
w.start(); w.session_open(); time.sleep(1.5); print('UP', flush=True); time.sleep(60)
"""
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "UP"
        other = RawDbWriter(db, ["KRW-BTC"])
        with pytest.raises(CollectorLockedError):
            other.start()
        # the live process's session must not be touched by the failed start
        assert _q(db, "SELECT COUNT(*) FROM collector_sessions WHERE disconnected_at IS NULL") == [(1,)]
    finally:
        p.kill()
        p.wait()
    w = RawDbWriter(db, ["KRW-BTC"])
    w.start()  # lock is free again; the killed process's session is closed
    w.stop()
    assert _q(db, "SELECT reason, disconnected_at IS NOT NULL FROM collector_sessions") == [("unclean_shutdown", 1)]


def test_collector_start_locked_returns_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from checkbit_ai import config
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "RAW_CSV", data / "btc_anomaly_dataset.csv")
    monkeypatch.setattr(config, "BACKUP_DIR", data / "backup")
    monkeypatch.setattr(config, "DB_PATH", data / "checkbit.sqlite3")
    monkeypatch.setattr(config, "STATE_PATH", data / "collector_state.json")
    monkeypatch.setattr(config, "LEGACY_CSV", tmp_path / "no_legacy.csv")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "no_model")
    monkeypatch.setattr(config, "REALTIME_ANALYSIS", False)
    monkeypatch.setattr(config, "AUTOSTART", "1")  # autostart must not crash startup when locked
    monkeypatch.setattr(config, "ADMIN_TOKEN", "t")
    holder = RawDbWriter(data / "raw.sqlite3", ["KRW-BTC"])
    holder.start()
    try:
        import main
        with TestClient(main.app) as client:
            assert client.get("/status").json()["running"] is False
            r = client.post("/collector/start", headers={"X-Admin-Token": "t"})
            assert r.status_code == 409 and r.json()["status"] == "locked"
    finally:
        holder.stop()
