"""Realtime feature service: a thread that finalizes each minute at t+60s+grace
by calling process_minutes() for every minute finalized since the last run."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..triggers.pipeline import process_triggers, trigger_range
from . import params as P
from . import store
from .pipeline import first_session_minute, last_final_minute, now_ms, open_raw, process_minutes

LOG = logging.getLogger("checkbit.featurepool")
# On first start with an empty feature DB, begin this far back (enough for the long window).
INITIAL_BACKFILL_MINUTES = P.LONG_WINDOW


class FeatureService:
    def __init__(self, raw_db: Path, features_db: Path, market: str, orderbook_interval_ms: int = 1000,
                 analyses_db: Path | None = None, triggers: bool = True) -> None:
        self.raw_db, self.features_db, self.market = Path(raw_db), Path(features_db), market
        self.analyses_db = Path(analyses_db) if analyses_db else None
        self.triggers = triggers
        self.orderbook_interval_ms = orderbook_interval_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.last_minute: int | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="featurepool", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def status(self) -> dict:
        return {"running": self._thread is not None and self._thread.is_alive(),
                "last_minute": self.last_minute, "last_error": self.last_error, "path": str(self.features_db)}

    def run_once(self) -> int:
        """Process every finalized minute after the last stored one."""
        if not self.raw_db.exists():
            return 0
        raw = open_raw(self.raw_db)
        feat = store.connect(self.features_db)
        try:
            now = now_ms()
            end = last_final_minute(now)
            last = store.last_minute(feat, self.market)
            if last is None:
                first = first_session_minute(raw, self.market)
                if first is None:
                    return 0
                start = max(first, end - INITIAL_BACKFILL_MINUTES)
            else:
                start = last + 1
            n = process_minutes(raw, feat, self.market, start, end, source="realtime", now=now,
                                orderbook_interval_ms=self.orderbook_interval_ms)
            self.last_minute = store.last_minute(feat, self.market)
            if self.triggers:  # detectors + candidates, LSTM_LAG_MINUTES behind the features
                rng = trigger_range(feat, self.market)
                if rng is not None:
                    process_triggers(feat, self.analyses_db, self.market, *rng)
            return n
        finally:
            raw.close()
            feat.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                n = self.run_once()
                if n > 1:
                    LOG.info("featurepool: processed %d minutes (caught up to %s)", n, self.last_minute)
                self.last_error = None
            except Exception as exc:  # keep the service alive; the next tick retries
                self.last_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("featurepool run failed")
            # sleep until the next minute is finalized
            now = now_ms()
            nxt = (last_final_minute(now) + 2) * P.MINUTE_MS + P.GRACE_MS
            self._stop.wait(max(1.0, (nxt - now) / 1000.0))
