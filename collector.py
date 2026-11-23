"""Upbit WebSocket collector for KRW-BTC trades + level-1 orderbook.

Improvements over the first version (same CSV columns + 2 appended):
- stop() really closes the WebSocket and joins the thread
- automatic reconnect with exponential backoff, ping/pong and a stale-feed watchdog
- only one collector thread can run (start() is idempotent)
- duplicate trades are skipped using Upbit's `sequential_id`
- CSV writes go through RawCsvStore (lock, partial-line repair, retry on disk errors)
- status() reports the *actual* connection state, not just a flag
- optionally (rawdb) also records every trade, full-depth orderbook snapshots and
  connection sessions in SQLite for the agent tools (checkbit_ai/rawdb.py)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import websocket

from checkbit_ai import config
from checkbit_ai.features import CSV_HEADER, CleanStats, Trade, parse_trade_row
from checkbit_ai.rawdb import CollectorLockedError, RawDbWriter, now_ms  # noqa: F401 (re-export)
from checkbit_ai.rawstore import RawCsvStore

LOG = logging.getLogger("checkbit.collector")

STALE_SECONDS = 45      # no message for this long while "connected" -> force reconnect
MAX_BACKOFF = 60

# Kept for backwards compatibility with the first version of this module.
FILE_NAME = str(config.RAW_CSV)


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None


class UpbitCollector:
    def __init__(self, store: RawCsvStore, market: str = config.MARKET, url: str = config.UPBIT_WS_URL,
                 on_trade: Callable[[Trade], None] | None = None, rawdb: RawDbWriter | None = None) -> None:
        self.store = store
        self.rawdb = rawdb
        # per-connection state, used to record why a session ended
        self._session_opened = False
        self._conn_error: str | None = None
        self._stale = False
        self.market = market
        self.url = url
        self.on_trade = on_trade
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocketApp | None = None
        self.orderbook: dict | None = None
        self._seen: set[str] = set()
        self._seen_order: deque[str] = deque()
        self.connected = False
        self.connected_since: float | None = None
        self.last_message_at: float | None = None
        self.last_trade_at: float | None = None
        self.started_at: float | None = None
        self.reconnects = 0
        self.session_rows = 0
        self.duplicates_skipped = 0
        self.message_errors = 0
        self.last_error: str | None = None
        self._load_recent_ids()

    # ------------------------------------------------------------ dedup
    def _remember(self, sid: str) -> bool:
        if sid in self._seen:
            return False
        self._seen.add(sid)
        self._seen_order.append(sid)
        if len(self._seen_order) > 50000:
            self._seen.discard(self._seen_order.popleft())
        return True

    def _load_recent_ids(self) -> None:
        try:
            for row in self.store.tail_rows(max_bytes=1 << 20):
                sid = (row.get("sequential_id") or "").strip()
                if sid:
                    self._remember(sid)
        except OSError as exc:
            LOG.warning("Could not read recent ids: %s", exc)

    # --------------------------------------------------------- callbacks
    def _on_open(self, ws) -> None:
        self.connected = True
        self.connected_since = time.time()
        self.last_message_at = time.time()
        LOG.info("Upbit WebSocket connected (%s)", self.market)
        ws.send(json.dumps([{"ticket": f"checkbit-{int(time.time())}"},
                            {"type": "trade", "codes": [self.market]},
                            {"type": "orderbook", "codes": [f"{self.market}.{config.ORDERBOOK_DEPTH}"]}]))
        if self.rawdb is not None:
            self.rawdb.session_open()
            self._session_opened = True

    def _on_message(self, ws, message) -> None:
        self.last_message_at = time.time()
        try:
            data = json.loads(message)
            kind = data.get("type")
            if self.rawdb is not None and kind in ("trade", "orderbook"):
                # before the CSV path's in-memory dedup: the DB dedups durably on its own
                self.rawdb.submit(data, now_ms())
            if kind == "orderbook":
                unit = data["orderbook_units"][0]
                self.orderbook = {"bid_price": unit["bid_price"], "ask_price": unit["ask_price"],
                                  "bid_size": unit["bid_size"], "ask_size": unit["ask_size"],
                                  "total_bid_size": data["total_bid_size"], "total_ask_size": data["total_ask_size"]}
            elif kind == "trade":
                self._handle_trade(data)
            elif "error" in data:
                self.last_error = f"Upbit 오류 응답: {data['error']}"
                LOG.error(self.last_error)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.message_errors += 1
            self.last_error = f"메시지 처리 오류: {exc}"
            LOG.warning("Bad message: %s", exc)

    def _handle_trade(self, data: dict) -> None:
        sid = str(data.get("sequential_id") or "")
        if sid and not self._remember(sid):
            self.duplicates_skipped += 1
            return
        ob = self.orderbook or {}
        row = [datetime.now(), data["code"], data["trade_price"], data["trade_volume"], data["ask_bid"],
               ob.get("bid_price", 0), ob.get("ask_price", 0), ob.get("bid_size", 0), ob.get("ask_size", 0),
               ob.get("total_bid_size", 0), ob.get("total_ask_size", 0), data.get("trade_timestamp", ""), sid]
        if self.store.append(row):
            self.session_rows += 1
        else:
            self.last_error = self.store.last_error
        self.last_trade_at = time.time()
        if self.on_trade is not None:
            try:
                rec = dict(zip(CSV_HEADER, row))
                rec["timestamp"] = str(row[0])
                trade = parse_trade_row(rec, self.market, CleanStats())
                if trade is not None:
                    self.on_trade(trade)
            except Exception as exc:  # never let analysis break collection
                LOG.exception("on_trade callback failed: %s", exc)

    def _on_error(self, ws, error) -> None:
        self.last_error = f"WebSocket 오류: {error}"
        self._conn_error = f"{type(error).__name__}: {error}"
        LOG.warning(self.last_error)

    def _on_close(self, ws, code, reason) -> None:
        self.connected = False
        LOG.info("WebSocket closed (code=%s reason=%s)", code, reason)

    # ------------------------------------------------------------- loops
    def _run(self) -> None:
        backoff = 1
        while not self._stop.is_set():
            ws = websocket.WebSocketApp(self.url, on_open=self._on_open, on_message=self._on_message,
                                        on_error=self._on_error, on_close=self._on_close)
            self._ws = ws
            began = time.time()
            self._session_opened, self._conn_error, self._stale = False, None, False
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                self.last_error = f"WebSocket 실행 오류: {exc}"
                self._conn_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("run_forever crashed")
            self._close_session()
            self.connected = False
            self.connected_since = None
            if self._stop.is_set():
                break
            if time.time() - began > 120:
                backoff = 1
            self.reconnects += 1
            LOG.info("Reconnecting in %ss (attempt %d)", backoff, self.reconnects)
            self._stop.wait(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        self._ws = None
        LOG.info("Collector thread finished")

    def _close_session(self) -> None:
        if self.rawdb is None or not self._session_opened:
            return
        if self._stop.is_set():
            reason = "stop"
        elif self._stale:
            reason = f"stale: no data for {STALE_SECONDS}s"
        elif self._conn_error:
            reason = f"error: {self._conn_error}"
        else:
            reason = "closed"
        self.rawdb.session_close(reason)
        self._session_opened = False

    def _watchdog(self) -> None:
        while not self._stop.wait(5):
            ws = self._ws
            if ws is not None and self.connected and self.last_message_at \
                    and time.time() - self.last_message_at > STALE_SECONDS:
                self.last_error = f"{STALE_SECONDS}초 동안 데이터 수신 없음 → 재연결"
                self._stale = True
                LOG.warning(self.last_error)
                try:
                    ws.close()
                except Exception:
                    pass

    # ---------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> bool:
        """Start collection; returns False if it was already running.

        Raises CollectorLockedError if another process is collecting into the same raw DB.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if not self._stop.is_set():
                    return False
                self._thread.join(timeout=10)  # a previous stop is still finishing
            if self.rawdb is not None:
                self.rawdb.start()  # takes the cross-process lock first
            self._stop = threading.Event()
            self.started_at = time.time()
            self.session_rows = 0
            self._thread = threading.Thread(target=self._run, name="upbit-collector", daemon=True)
            self._thread.start()
            threading.Thread(target=self._watchdog, name="collector-watchdog", daemon=True).start()
            LOG.info("Collector started")
            return True

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop collection and close the socket; returns True when the thread ended."""
        with self._lock:
            self._stop.set()
            ws = self._ws
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        if self.rawdb is not None:
            self.rawdb.stop()  # drains queued rows and session events, releases the lock
        self.store.close()
        self.connected = False
        ended = t is None or not t.is_alive()
        LOG.info("Collector stopped (thread ended=%s)", ended)
        return ended

    def status(self) -> dict:
        now = time.time()
        receiving = bool(self.connected and self.last_message_at and now - self.last_message_at < 15)
        return {"running": self.running, "connected": self.connected, "receiving": receiving,
                "market": self.market, "rows": self.store.rows, "session_rows": self.session_rows,
                "started_at": _iso(self.started_at), "connected_since": _iso(self.connected_since),
                "last_message_at": _iso(self.last_message_at), "last_trade_at": _iso(self.last_trade_at),
                "orderbook_ready": self.orderbook is not None, "reconnects": self.reconnects,
                "duplicates_skipped": self.duplicates_skipped, "message_errors": self.message_errors,
                "write_errors": self.store.write_errors, "pending_rows": self.store.pending,
                "last_error": self.last_error,
                "raw_db": self.rawdb.status() if self.rawdb is not None else None}


def start_collector() -> None:
    """Backwards compatible blocking entry point (python -c 'import collector; ...')."""
    config.ensure_dirs()
    c = UpbitCollector(RawCsvStore(config.RAW_CSV, config.BACKUP_DIR, [config.LEGACY_CSV]))
    c.start()
    try:
        while c.running:
            time.sleep(1)
    except KeyboardInterrupt:
        c.stop()
