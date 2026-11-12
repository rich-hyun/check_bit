import json
import threading
import time

import pytest
from websockets.sync.server import serve

from checkbit_ai.features import CSV_HEADER, LEGACY_HEADER
from checkbit_ai.rawstore import RawCsvStore, count_rows
from checkbit_ai.store import Store
from collector import UpbitCollector


def _legacy(path, n=3):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(LEGACY_HEADER) + "\n")
        for i in range(n):
            f.write(f"2026-09-08 13:41:5{i}.0,KRW-BTC,106979000.0,0.001,BID,106979000.0,106980000.0,0.1,0.1,1.7,7.0\n")


def test_legacy_migration_keeps_original(tmp_path):
    legacy = tmp_path / "btc_anomaly_dataset.csv"
    _legacy(legacy)
    raw = RawCsvStore(tmp_path / "data" / "raw.csv", tmp_path / "data" / "backup", [legacy])
    assert raw.rows == 3 and legacy.exists()
    assert (tmp_path / "data" / "raw.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_HEADER)


def test_legacy_header_in_place_is_converted_with_backup(tmp_path):
    p = tmp_path / "raw.csv"
    _legacy(p, 2)
    raw = RawCsvStore(p, tmp_path / "bk")
    assert raw.rows == 2 and any("legacy" in x.name for x in (tmp_path / "bk").iterdir())


def test_partial_last_line_is_repaired(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text(",".join(CSV_HEADER) + "\n" + "a," * 12 + "b\n" + "2026-09-08,KRW-B", encoding="utf-8")
    raw = RawCsvStore(p, tmp_path / "bk")
    assert raw.rows == 1 and p.read_text(encoding="utf-8").endswith("b\n")
    assert "KRW-B" in (tmp_path / "bk" / "corrupt_fragments.log").read_text(encoding="utf-8")


def test_unknown_header_moved_aside(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    raw = RawCsvStore(p, tmp_path / "bk")
    assert raw.rows == 0 and len(list((tmp_path / "bk").iterdir())) == 1


def test_concurrent_append_and_snapshot(tmp_path):
    raw = RawCsvStore(tmp_path / "raw.csv", tmp_path / "bk")
    row = ["2026-11-12 10:00:00", "KRW-BTC", 1, 1, "BID", 1, 2, 1, 1, 1, 1, 1, "1"]
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            raw.append(row)
    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(20):
            data = b"".join(raw.iter_snapshot())
            assert data.endswith(b"\n")  # never a half-written row
    finally:
        stop.set()
        t.join()
    assert count_rows(raw.path) == raw.rows


# ---------------------------------------------------------------- collector
class FakeUpbit:
    """Local WebSocket server imitating Upbit (binary JSON frames)."""

    def __init__(self, drop_after: int | None = None) -> None:
        self.connections = 0
        self.open_now = 0
        self.drop_after = drop_after
        self.server = serve(self.handler, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(self, ws):
        self.connections += 1
        self.open_now += 1
        conn = self.connections
        try:
            json.loads(ws.recv())  # subscription
            ws.send(json.dumps({"type": "orderbook", "code": "KRW-BTC", "total_bid_size": 3.0, "total_ask_size": 4.0,
                                "orderbook_units": [{"bid_price": 99999000, "ask_price": 100000000,
                                                     "bid_size": 0.5, "ask_size": 0.6}]}).encode())
            for i in range(1, 6):
                sid = conn * 100 + i
                msg = {"type": "trade", "code": "KRW-BTC", "trade_price": 100000000, "trade_volume": 0.01,
                       "ask_bid": "BID", "trade_timestamp": int(time.time() * 1000), "sequential_id": sid}
                ws.send(json.dumps(msg).encode())
                if i == 2:
                    ws.send(json.dumps(msg).encode())  # duplicate delivery
            if self.drop_after is not None and conn <= self.drop_after:
                return  # server drops the connection -> client must reconnect
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


def test_collector_start_stop_dedup(tmp_path):
    fake = FakeUpbit()
    got = []
    raw = RawCsvStore(tmp_path / "raw.csv", tmp_path / "bk")
    c = UpbitCollector(raw, url=fake.url, on_trade=got.append)
    try:
        assert c.start() is True
        assert c.start() is False  # no duplicate collector thread
        assert _wait(lambda: raw.rows == 5)
        st = c.status()
        assert st["running"] and st["connected"] and st["duplicates_skipped"] == 1 and st["orderbook_ready"]
        assert len(got) == 5 and got[0].has_book
        assert c.stop() is True
        assert _wait(lambda: fake.open_now == 0)  # the socket was really closed
        assert c.status()["running"] is False and c.status()["connected"] is False
        n = raw.rows
        time.sleep(0.5)
        assert raw.rows == n
        # Restart after stop works and does not re-store ids already in the CSV
        assert c.start() is True
        assert _wait(lambda: raw.rows == 10)
        c.stop()
    finally:
        c.stop()
        fake.close()
    lines = (tmp_path / "raw.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(CSV_HEADER) and len(lines) == 11


def test_collector_reconnects_after_drop(tmp_path):
    fake = FakeUpbit(drop_after=1)
    raw = RawCsvStore(tmp_path / "raw.csv", tmp_path / "bk")
    c = UpbitCollector(raw, url=fake.url)
    try:
        c.start()
        assert _wait(lambda: fake.connections >= 2 and raw.rows == 10, timeout=15)
        assert c.status()["reconnects"] >= 1 and c.status()["connected"]
    finally:
        c.stop()
        fake.close()


def test_store_history_feedback_and_summary(tmp_path):
    s = Store(tmp_path / "db.sqlite3")
    base = {"market": "KRW-BTC", "threshold": 1.0, "model_version": "v1", "close": 1.0, "features": {"a": 1},
            "explanation": {"summary": "s", "top_features": [{"feature": "a", "label": "A", "contribution_pct": 50}]},
            "source": "manual"}
    a = s.save_analysis({**base, "minute": 10, "detected_at": "t10", "score": 2.0, "is_anomaly": True})
    b = s.save_analysis({**base, "minute": 11, "detected_at": "t11", "score": 0.5, "is_anomaly": False})
    assert s.save_analysis({**base, "minute": 10, "detected_at": "t10", "score": 2.0, "is_anomaly": True}) == a
    assert [x["id"] for x in s.list_analyses("KRW-BTC", anomalies_only=True)] == [a]
    s.add_feedback(a, "normal", None)
    s.add_feedback(a, "anomaly", "확인함")  # latest label wins
    s.add_feedback(b, "hold", None)
    assert s.add_feedback(999, "normal", None) is None
    d = s.get_analysis(a)
    assert d["user_label"] == "anomaly" and len(d["feedback"]) == 2
    summ = s.feedback_summary("KRW-BTC")
    assert summ["reviewed"] == 2 and summ["decided"] == 1 and summ["agreement_rate"] == 1.0
    with pytest.raises(Exception):
        s._db.execute("INSERT INTO feedback (analysis_id,label,created_at) VALUES (?,?,?)", (a, "bad", "x"))
