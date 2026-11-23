"""Raw market data in SQLite: every trade, full-depth orderbook snapshots, and
collector connection sessions (coverage).

This is written *in addition to* the trade CSV (RawCsvStore), which the LSTM AE
pipeline keeps using. The agent tools query this DB because they need random
access to past intervals, all orderbook levels, and to tell "no trades" apart
from "collector was not connected".

- All market times are Upbit epoch ms (UTC). `received_at` and session times
  are server epoch ms.
- trades: PK (market, sequential_id) + INSERT OR IGNORE, so dedup is durable
  across restarts (unlike the collector's in-memory id set).
- orderbook snapshots are sampled per market by Upbit timestamp
  (ORDERBOOK_INTERVAL_MS, 0 = keep all). A snapshot ignored as a duplicate is
  detected with changes() and its levels are skipped.
- One writer thread owns the connection and commits in batches.
- A non-blocking flock on `<db>.lock` keeps a second process (e.g. another
  uvicorn worker) from collecting into the same DB.
"""
from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: no cross-process guard
    fcntl = None

LOG = logging.getLogger("checkbit.rawdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    market             TEXT    NOT NULL,
    sequential_id      INTEGER NOT NULL,
    trade_timestamp    INTEGER NOT NULL,  -- Upbit trade time (epoch ms, UTC)
    trade_price        REAL    NOT NULL,
    trade_volume       REAL    NOT NULL,
    ask_bid            TEXT    NOT NULL,  -- ASK (seller-initiated) / BID (buyer-initiated)
    prev_closing_price REAL,
    stream_type        TEXT,              -- SNAPSHOT / REALTIME
    received_at        INTEGER NOT NULL,  -- server receive time (epoch ms)
    PRIMARY KEY (market, sequential_id)
);
CREATE INDEX IF NOT EXISTS idx_trades_market_ts ON trades(market, trade_timestamp);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id             INTEGER PRIMARY KEY,
    market         TEXT    NOT NULL,
    timestamp      INTEGER NOT NULL,      -- Upbit orderbook time (epoch ms, UTC)
    total_ask_size REAL    NOT NULL,
    total_bid_size REAL    NOT NULL,
    level          REAL,                  -- price grouping unit (0 = none)
    stream_type    TEXT,
    received_at    INTEGER NOT NULL,
    UNIQUE (market, timestamp)
);

CREATE TABLE IF NOT EXISTS orderbook_units (
    snapshot_id INTEGER NOT NULL REFERENCES orderbook_snapshots(id),
    depth       INTEGER NOT NULL,         -- 0 = best bid/ask
    ask_price   REAL NOT NULL,
    bid_price   REAL NOT NULL,
    ask_size    REAL NOT NULL,
    bid_size    REAL NOT NULL,
    PRIMARY KEY (snapshot_id, depth)
) WITHOUT ROWID;

-- Coverage: time outside these intervals is a collection gap.
CREATE TABLE IF NOT EXISTS collector_sessions (
    id              INTEGER PRIMARY KEY,
    market          TEXT    NOT NULL,
    connected_at    INTEGER NOT NULL,     -- subscription sent (server epoch ms)
    disconnected_at INTEGER,              -- NULL = in progress
    reason          TEXT                  -- stop / stale / error: ... / closed ... / unclean_shutdown
);
CREATE INDEX IF NOT EXISTS idx_sessions_market ON collector_sessions(market, connected_at);
"""

BATCH_MAX = 500
BATCH_SECONDS = 1.0
QUEUE_MAX = 100_000
_STOP = object()
_SESSION_OPEN = "session_open"
_SESSION_CLOSE = "session_close"


class CollectorLockedError(RuntimeError):
    pass


def now_ms() -> int:
    return int(time.time() * 1000)


def connect(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db(path: Path) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------- writes
def insert_trades(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """Returns the number of rows actually inserted (duplicates are ignored)."""
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO trades
           (market, sequential_id, trade_timestamp, trade_price, trade_volume,
            ask_bid, prev_closing_price, stream_type, received_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return conn.total_changes - before


def insert_orderbook(conn: sqlite3.Connection, snap: tuple, units: list[tuple]) -> bool:
    """Insert one snapshot. Returns False (and writes no levels) on a duplicate."""
    cur = conn.execute(
        """INSERT OR IGNORE INTO orderbook_snapshots
           (market, timestamp, total_ask_size, total_bid_size, level, stream_type, received_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        snap,
    )
    # lastrowid is not reliable after an ignored insert; ask SQLite directly.
    if conn.execute("SELECT changes()").fetchone()[0] == 0:
        return False
    sid = cur.lastrowid
    conn.executemany(
        """INSERT INTO orderbook_units
           (snapshot_id, depth, ask_price, bid_price, ask_size, bid_size)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(sid, *u) for u in units],
    )
    return True


def open_sessions(conn: sqlite3.Connection, markets: list[str], connected_at: int) -> None:
    conn.executemany("INSERT INTO collector_sessions (market, connected_at) VALUES (?, ?)",
                     [(m, connected_at) for m in markets])


def close_open_sessions(conn: sqlite3.Connection, disconnected_at: int, reason: str) -> int:
    """Close every open session. The caller must hold the DB lock (all open rows are its own)."""
    cur = conn.execute("UPDATE collector_sessions SET disconnected_at = ?, reason = ?"
                       " WHERE disconnected_at IS NULL", (disconnected_at, reason))
    return cur.rowcount


def close_dangling_sessions(conn: sqlite3.Connection) -> int:
    """Close sessions left open by a killed process at the last data received in them.

    Call only while holding the DB lock, so another live process's session is never closed.
    """
    cur = conn.execute(
        """UPDATE collector_sessions SET
               reason = 'unclean_shutdown',
               disconnected_at = MAX(
                 connected_at,
                 COALESCE((SELECT MAX(t.received_at) FROM trades t
                            WHERE t.market = collector_sessions.market
                              AND t.received_at >= collector_sessions.connected_at), 0),
                 COALESCE((SELECT MAX(o.received_at) FROM orderbook_snapshots o
                            WHERE o.market = collector_sessions.market
                              AND o.received_at >= collector_sessions.connected_at), 0))
           WHERE disconnected_at IS NULL""")
    return cur.rowcount


# ----------------------------------------------------------------- parsing
def _trade_row(d: dict, received_at: int) -> tuple | None:
    try:
        return (str(d["code"]), int(d["sequential_id"]), int(d["trade_timestamp"]),
                float(d["trade_price"]), float(d["trade_volume"]), str(d["ask_bid"]),
                d.get("prev_closing_price"), d.get("stream_type"), received_at)
    except (KeyError, TypeError, ValueError):
        return None


def _orderbook(d: dict, received_at: int) -> tuple[tuple, list[tuple]] | None:
    try:
        snap = (str(d["code"]), int(d["timestamp"]), float(d["total_ask_size"]),
                float(d["total_bid_size"]), d.get("level"), d.get("stream_type"), received_at)
        units = [(i, float(u["ask_price"]), float(u["bid_price"]), float(u["ask_size"]), float(u["bid_size"]))
                 for i, u in enumerate(d["orderbook_units"])]
    except (KeyError, TypeError, ValueError):
        return None
    return (snap, units) if units else None


# ------------------------------------------------------------------ writer
class RawDbWriter:
    """Queue + single writer thread. Thread-safe submit(); start()/stop() are not
    re-entrant and are called by the collector under its own lock."""

    def __init__(self, path: Path, markets: list[str], orderbook_interval_ms: int = 1000) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.markets = list(markets)
        self.orderbook_interval_ms = max(0, int(orderbook_interval_ms))
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._lock_fd: int | None = None
        self._last_saved_ob: dict[str, int] = {}
        self._stats_lock = threading.Lock()
        init_db(self.path)
        conn = connect(self.path)
        try:
            trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            snaps = conn.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0]
        finally:
            conn.close()
        self._stats = {"trades": trades, "orderbook_snapshots": snaps, "duplicate_trades": 0,
                       "duplicate_snapshots": 0, "sampled_out": 0, "skipped_invalid": 0,
                       "queue_dropped": 0, "write_errors": 0, "last_error": None}

    # -------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._acquire_lock()
        try:
            conn = connect(self.path)
            try:
                with conn:
                    n = close_dangling_sessions(conn)
            finally:
                conn.close()
            if n:
                LOG.warning("Closed %d dangling collector session(s) as unclean_shutdown", n)
        except Exception:
            self._release_lock()
            raise
        self._queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread = threading.Thread(target=self._loop, name="rawdb-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Drain the queue, close open sessions, release the lock."""
        t = self._thread
        if t is not None:
            self._queue.put(_STOP)
            t.join(timeout=timeout)
            if t.is_alive():
                LOG.error("rawdb writer did not stop within %.0fs", timeout)
        self._thread = None
        self._release_lock()

    # --------------------------------------------------------------- inputs
    def submit(self, msg: dict, received_at: int | None = None) -> None:
        if not self.running:
            return
        try:
            self._queue.put_nowait((msg, received_at or now_ms()))
        except queue.Full:
            self._incr("queue_dropped")

    def session_open(self, ts: int | None = None) -> None:
        if self.running:
            self._queue.put((_SESSION_OPEN, ts or now_ms()))  # never dropped

    def session_close(self, reason: str, ts: int | None = None) -> None:
        if self.running:
            self._queue.put((_SESSION_CLOSE, ts or now_ms(), reason[:500]))

    def status(self) -> dict:
        with self._stats_lock:
            s = dict(self._stats)
        s.update(running=self.running, path=str(self.path), orderbook_interval_ms=self.orderbook_interval_ms,
                 queue=self._queue.qsize())
        return s

    # ---------------------------------------------------------------- locks
    def _acquire_lock(self) -> None:
        if fcntl is None:
            LOG.warning("fcntl unavailable: cross-process collector guard disabled")
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise CollectorLockedError(f"다른 프로세스가 이미 수집 중입니다 (lock: {self.lock_path})")
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    # --------------------------------------------------------------- writer
    def _loop(self) -> None:
        conn = connect(self.path)
        try:
            stopping = False
            while not stopping:
                batch = []
                deadline = time.monotonic() + BATCH_SECONDS
                while len(batch) < BATCH_MAX:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is _STOP:
                        stopping = True
                        break
                    batch.append(item)
                if stopping:  # drain what is left after the stop signal
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not _STOP:
                            batch.append(item)
                if batch:
                    self._write(conn, batch)
        finally:
            try:  # sessions whose close event never arrived (should not happen)
                with conn:
                    close_open_sessions(conn, now_ms(), "stop")
            except Exception:
                LOG.exception("closing open sessions on writer exit failed")
            conn.close()

    def _write(self, conn: sqlite3.Connection, batch: list) -> None:
        controls = [b for b in batch if b[0] in (_SESSION_OPEN, _SESSION_CLOSE)]
        if controls:
            # Own transaction: coverage is recorded even if the data batch fails.
            try:
                with conn:
                    for c in controls:
                        if c[0] == _SESSION_OPEN:
                            close_open_sessions(conn, c[1], "superseded")  # not expected
                            open_sessions(conn, self.markets, c[1])
                        else:
                            close_open_sessions(conn, c[1], c[2])
            except Exception as exc:
                LOG.exception("session write failed")
                self._set(last_error=f"session write: {type(exc).__name__}: {exc}")

        trades, books, invalid = [], [], 0
        for item in batch:
            if item[0] in (_SESSION_OPEN, _SESSION_CLOSE):
                continue
            msg, received_at = item
            kind = msg.get("type")
            if kind == "trade":
                row = _trade_row(msg, received_at)
                if row is None:
                    invalid += 1
                else:
                    trades.append(row)
            elif kind == "orderbook":
                ob = _orderbook(msg, received_at)
                if ob is None:
                    invalid += 1
                else:
                    books.append(ob)

        sampled = ob_new = ob_dup = 0
        saved = dict(self._last_saved_ob)
        try:
            with conn:
                inserted = insert_trades(conn, trades)
                for snap, units in books:
                    market, ts = snap[0], snap[1]
                    prev = saved.get(market)
                    if self.orderbook_interval_ms and prev is not None and ts - prev < self.orderbook_interval_ms:
                        sampled += 1
                        continue
                    if insert_orderbook(conn, snap, units):
                        ob_new += 1
                        saved[market] = ts
                    else:
                        ob_dup += 1
        except Exception as exc:
            LOG.exception("raw db batch write failed (%d items dropped)", len(batch))
            with self._stats_lock:
                self._stats["write_errors"] += 1
                self._stats["skipped_invalid"] += invalid
                self._stats["last_error"] = f"write: {type(exc).__name__}: {exc}"
            return
        self._last_saved_ob = saved
        with self._stats_lock:
            s = self._stats
            s["trades"] += inserted
            s["duplicate_trades"] += len(trades) - inserted
            s["orderbook_snapshots"] += ob_new
            s["duplicate_snapshots"] += ob_dup
            s["sampled_out"] += sampled
            s["skipped_invalid"] += invalid

    def _set(self, **kw) -> None:
        with self._stats_lock:
            self._stats.update(kw)

    def _incr(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] += n
