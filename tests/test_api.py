"""End-to-end API tests with SYNTHETIC data written in the collector CSV format."""
import shutil
from contextlib import contextmanager
import time

import pytest
from fastapi.testclient import TestClient

from checkbit_ai import config
from checkbit_ai.synthetic import generate
from conftest import retry_winsock


@pytest.fixture()
def env(tmp_path, monkeypatch, trained_model):
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "RAW_CSV", data / "btc_anomaly_dataset.csv")
    monkeypatch.setattr(config, "BACKUP_DIR", data / "backup")
    monkeypatch.setattr(config, "DB_PATH", data / "checkbit.sqlite3")
    monkeypatch.setattr(config, "STATE_PATH", data / "collector_state.json")
    monkeypatch.setattr(config, "LEGACY_CSV", tmp_path / "no_legacy.csv")
    monkeypatch.setattr(config, "AUTOSTART", "0")
    monkeypatch.setattr(config, "ADMIN_TOKEN", "test-token")
    model_dir = tmp_path / "model"
    shutil.copytree(trained_model, model_dir)
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    return tmp_path


@contextmanager
def _client():
    """TestClient with a retry for a Windows-only flake: creating the asyncio
    self-pipe (socket.socketpair) occasionally fails with WinError 10014 before
    the app's lifespan even starts. Production runs on Linux."""
    import main
    client = TestClient(main.app)
    retry_winsock(client.__enter__)
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


def _write_recent(env, minutes=120, anomalies=(110,)):
    start = (time.time() // 60 - minutes) * 60  # ends at the current minute
    generate(config.RAW_CSV, minutes=minutes, start_ts=start, anomalies=anomalies)


def test_model_not_ready_is_explicit(env, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", env / "missing")
    with _client() as c:
        st = c.get("/ai/status").json()
        assert st["model_loaded"] is False and "모델 파일이 없습니다" in st["model_error"]
        r = c.post("/ai/analyze").json()
        assert r["status"] == "model_not_ready" and "score" not in r
        assert c.get("/ai/analysis/latest").json()["status"] == "model_not_ready"
        tl = c.get("/ai/timeline").json()
        assert tl["model_version"] is None


def test_insufficient_data(env):
    with _client() as c:
        r = c.post("/ai/analyze").json()
        assert r["status"] == "insufficient_data" and r["bars"] == 0


def test_full_flow_with_synthetic_data(env):
    _write_recent(env)
    with _client() as c:
        assert c.get("/").status_code == 200
        st = c.get("/status").json()
        assert {"running", "market", "rows"} <= st.keys() and st["rows"] > 0 and st["running"] is False
        r = c.post("/ai/analyze").json()
        assert r["status"] == "ok" and r["model_version"].startswith("lstm-ae-")
        assert r["threshold"] > 0 and isinstance(r["is_anomaly"], bool) and r["explanation"]["evidence"]
        assert set(r["features"]) >= {"log_return", "spread_bps", "buy_sell_imbalance"}
        # backfill runs in the background on startup
        deadline = time.time() + 10
        while time.time() < deadline and c.get("/ai/status").json()["backfill"] != "done":
            time.sleep(0.2)
        tl = c.get("/ai/timeline?minutes=120").json()
        scored = [p for p in tl["points"] if p["score"] is not None]
        assert len(scored) >= 50 and tl["threshold"] == r["threshold"]
        det = c.get("/ai/detections?limit=50").json()
        assert det["count"] >= 1 and all(i["is_anomaly"] for i in det["items"])
        aid = det["items"][0]["id"]
        detail = c.get(f"/ai/detections/{aid}").json()
        assert detail["explanation"]["top_features"] and "투자 조언" in detail["explanation"]["disclaimer"]
        assert c.post(f"/ai/detections/{aid}/feedback", json={"label": "anomaly", "note": "급등 확인"}).status_code == 200
        assert c.post(f"/ai/detections/{aid}/feedback", json={"label": "maybe"}).status_code == 422
        assert c.post("/ai/detections/999999/feedback", json={"label": "normal"}).status_code == 404
        assert c.get(f"/ai/detections/{aid}").json()["user_label"] == "anomaly"
        summ = c.get("/ai/feedback/summary").json()
        assert summ["reviewed"] == 1 and summ["user_confirmed_precision"] == 1.0
        latest = c.get("/ai/analysis/latest").json()
        assert latest["status"] == "ok"
        dl = c.get("/download")
        assert dl.status_code == 200 and dl.text.startswith("timestamp,market,trade_price")


def test_burst_minute_is_flagged(env):
    _write_recent(env, minutes=120, anomalies=(110,))
    with _client() as c:
        deadline = time.time() + 10
        while time.time() < deadline and c.get("/ai/status").json()["backfill"] != "done":
            time.sleep(0.2)
        det = c.get("/ai/detections?limit=100").json()["items"]
        burst_minute = int(time.time() // 60) - 120 + 110
        assert any(abs(d["minute"] - burst_minute) <= 2 for d in det), (burst_minute, [d["minute"] for d in det], c.get("/ai/status").json())


def test_admin_endpoints_require_token(env, monkeypatch):
    with _client() as c:
        assert c.post("/collector/start").status_code == 401
        assert c.post("/collector/stop", headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert c.post("/ai/model/reload").status_code == 401
        r = c.post("/ai/model/reload", headers={"X-Admin-Token": "test-token"}).json()
        assert r["loaded"] is True
        r = c.post("/collector/stop", headers={"X-Admin-Token": "test-token"}).json()
        assert r["status"] == "stopped"
        assert config.STATE_PATH.exists()
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    with _client() as c:
        assert c.post("/collector/start", headers={"X-Admin-Token": "x"}).status_code == 403


def test_restart_restores_desired_state(env, monkeypatch):
    import main
    monkeypatch.setattr(config, "AUTOSTART", None)
    config.ensure_dirs()
    config.STATE_PATH.write_text('{"running": true}', encoding="utf-8")
    assert main._should_autostart() is True
    config.STATE_PATH.write_text('{"running": false}', encoding="utf-8")
    assert main._should_autostart() is False


def test_first_start_imports_predeploy_backup(env):
    """Simulates the first production start after this change: the old root CSV
    was backed up by deploy.yml and a header-less fragment was re-created."""
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bk = config.BACKUP_DIR / "btc_anomaly_dataset_predeploy_20261112_000000.csv"
    bk.write_text("timestamp,market,trade_price,trade_volume,ask_bid,bid_price,ask_price,bid_size,ask_size,"
                  "total_bid_size,total_ask_size\n"
                  "2026-09-08 13:41:59.98,KRW-BTC,106979000.0,0.008,BID,106979000.0,106980000.0,0.08,0.01,1.7,7.0\n",
                  encoding="utf-8")
    config.LEGACY_CSV.write_text("2026-11-12 10:00:00,KRW-BTC,1,1,BID,0,0,0,0,0,0\n", encoding="utf-8")
    with _client() as c:
        assert c.get("/status").json()["rows"] == 1
    assert bk.exists()
