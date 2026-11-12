"""Realtime inference service: minute bars buffer -> features -> LSTM AE score.

Runs inside the FastAPI process. Inference only (NumPy, ~ms per window);
training is never triggered from here.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .explain import explain_window
from .features import (FEATURES, FEATURE_LABELS, CleanStats, MinuteAggregator, MinuteBar, Scaler, Trade,
                       aggregate_minutes, build_features, minute_iso, parse_trade_row)
from .lstm_ae import LSTMAutoEncoder, score_from_errors
from .rawstore import RawCsvStore
from .store import Store

LOG = logging.getLogger("checkbit.ai")
STALE_MINUTES = 3


class NotReady(Exception):
    def __init__(self, code: str, message: str, **extra) -> None:
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def as_dict(self) -> dict:
        return {"status": self.code, "message": self.message, **self.extra}


def _utc(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat()


class DetectionService:
    def __init__(self, store: Store, raw: RawCsvStore, model_dir: Path, market: str = "KRW-BTC",
                 buffer_minutes: int = 720, realtime: bool = True) -> None:
        self.store, self.raw, self.model_dir, self.market = store, raw, Path(model_dir), market
        self.buffer: deque[MinuteBar] = deque(maxlen=buffer_minutes)
        self.agg = MinuteAggregator()
        self.realtime = realtime
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.model: LSTMAutoEncoder | None = None
        self.meta: dict | None = None
        self.scaler: Scaler | None = None
        self.model_error: str | None = None
        self.report: dict | None = None
        self.model_loaded_at: str | None = None
        self.last_analysis_at: str | None = None
        self.last_analyzed_minute: int | None = None
        self.last_result: dict | None = None
        self.last_error: str | None = None
        self.bootstrap_info: dict = {}
        self.backfill_state: str = "idle"  # idle | running | done

    # ------------------------------------------------------------- model
    def load_model(self) -> bool:
        npz, meta_path = self.model_dir / "model.npz", self.model_dir / "meta.json"
        try:
            if not npz.exists() or not meta_path.exists():
                raise FileNotFoundError(f"모델 파일이 없습니다 ({self.model_dir.name}/model.npz, meta.json). "
                                        "오프라인 학습 후 배포해야 합니다.")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("features") != FEATURES:
                raise ValueError("모델의 특징량 목록이 현재 전처리 코드와 다릅니다. 모델을 다시 학습하세요.")
            model = LSTMAutoEncoder.load(npz)
            if model.window != meta["window"] or model.n_features != len(FEATURES):
                raise ValueError("모델 파일과 메타데이터의 구조가 일치하지 않습니다.")
            if not (isinstance(meta.get("threshold"), (int, float)) and math.isfinite(meta["threshold"])
                    and meta["threshold"] > 0):
                raise ValueError("임계값이 올바르지 않습니다.")
            scaler = Scaler.from_dict(meta["scaler"])
        except Exception as exc:  # report, do not crash the API
            with self._lock:
                self.model, self.meta, self.scaler = None, None, None
                self.model_error = str(exc)
            LOG.warning("Model not loaded: %s", exc)
            return False
        report = None
        try:
            report = json.loads((self.model_dir / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        with self._lock:
            self.model, self.meta, self.scaler, self.report = model, meta, scaler, report
            self.model_error = None
            self.model_loaded_at = _utc()
        LOG.info("Loaded model %s (window=%s, threshold=%.5f)", meta["version"], meta["window"], meta["threshold"])
        return True

    @property
    def model_version(self) -> str | None:
        return self.meta["version"] if self.meta else None

    # ------------------------------------------------------------ buffer
    def bootstrap(self, max_bytes: int = 16 << 20) -> None:
        """Fill the bar buffer from the end of the collected CSV (after restart)."""
        stats = CleanStats()
        trades = []
        for row in self.raw.tail_rows(max_bytes=max_bytes):
            t = parse_trade_row(row, self.market, stats)
            if t is not None:
                trades.append(t)
        trades.sort(key=lambda t: t.ts)
        bars = aggregate_minutes(trades)
        now_minute = int(time.time() // 60)
        with self._lock:
            self.buffer.clear()
            for b in bars:
                if b.minute < now_minute:
                    self.buffer.append(b)
                else:
                    self.agg.current = b  # minute still in progress
        self.bootstrap_info = {"trades": len(trades), "bars": len(bars), "clean": stats.as_dict()}
        LOG.info("Bootstrapped %d bars from %d trades", len(bars), len(trades))

    def on_trade(self, trade: Trade) -> None:
        with self._lock:
            done = self.agg.add(trade)
            self._append(done)
        if done:
            self._wake.set()

    def _append(self, bars: list[MinuteBar]) -> None:
        for b in bars:
            if self.buffer and b.minute <= self.buffer[-1].minute:
                continue
            self.buffer.append(b)

    def _bars(self) -> list[MinuteBar]:
        with self._lock:
            self._append(self.agg.flush_due(time.time()))
            return list(self.buffer)

    # ---------------------------------------------------------- analysis
    def _require_model(self):
        with self._lock:
            if self.model is None:
                raise NotReady("model_not_ready", "학습된 이상탐지 모델이 준비되지 않았습니다. 분석을 수행하지 않았습니다.",
                               detail=self.model_error)
            return self.model, self.meta, self.scaler

    def _score_windows(self, frame, ends: np.ndarray, model, meta, scaler, source: str, explain_all=True) -> list[dict]:
        W = meta["window"]
        Xs = scaler.transform(np.nan_to_num(frame.X, nan=0.0))
        idx = ends[:, None] + np.arange(-W + 1, 1)[None, :]
        err = model.errors(Xs[idx])
        scores = score_from_errors(err, meta.get("score_steps"))
        out = []
        for k, e in enumerate(ends):
            score = float(scores[k])
            is_anom = score > meta["threshold"]
            rows = idx[k]
            explanation = explain_window(minutes=frame.minutes[rows], raw=frame.X[rows], err=err[k],
                                         score=score, threshold=meta["threshold"], meta=meta,
                                         close=frame.close[rows], value=frame.value[rows]) \
                if (explain_all or is_anom) else None
            rec = {"market": self.market, "minute": int(frame.minutes[e]), "detected_at": minute_iso(int(frame.minutes[e])),
                   "score": score, "threshold": float(meta["threshold"]), "is_anomaly": bool(is_anom),
                   "model_version": meta["version"], "close": float(frame.close[e]),
                   "features": frame.row_features(int(e)), "explanation": explanation, "source": source}
            rec["id"] = self.store.save_analysis(rec)
            out.append(rec)
        return out

    def analyze_latest(self, source: str = "manual") -> dict:
        model, meta, scaler = self._require_model()
        W = meta["window"]
        bars = self._bars()
        need = W + 10 + 1  # window + rolling-volatility warm-up + first return
        if len(bars) < need:
            raise NotReady("insufficient_data", f"분석에 필요한 1분 데이터가 부족합니다 ({len(bars)}/{need}분). "
                           "데이터 수집을 계속해 주세요.", bars=len(bars), required=need)
        frame = build_features(bars[-(need + 120):])
        last = len(frame) - 1
        lo = last - W + 1
        if lo < 0 or not frame.valid[lo:].all() or len(set(frame.segments[lo:].tolist())) != 1:
            raise NotReady("insufficient_data", "최근 구간에 끊김이나 결측이 있어 연속된 "
                           f"{W}분 윈도우를 만들 수 없습니다. 수집이 이어지면 자동으로 분석됩니다.",
                           bars=len(bars), required=need)
        with self._analysis_lock:
            rec = self._score_windows(frame, np.array([last]), model, meta, scaler, source)[0]
        now_minute = int(time.time() // 60)
        stale = now_minute - rec["minute"] > STALE_MINUTES
        result = {"status": "ok", **self._public(rec), "stale": stale,
                  "message": ("최신 수집 데이터가 아닙니다. 데이터 수집 상태를 확인하세요." if stale else None)}
        with self._lock:
            self.last_result, self.last_analysis_at = result, _utc()
            self.last_analyzed_minute = rec["minute"]
        return result

    def backfill(self, max_windows: int = 720) -> int:
        """Score buffered windows that have no stored result yet (e.g. after restart)."""
        try:
            model, meta, scaler = self._require_model()
        except NotReady:
            self.backfill_state = "done"
            return 0
        self.backfill_state = "running"
        try:
            return self._backfill(model, meta, scaler, max_windows)
        finally:
            self.backfill_state = "done"

    def _backfill(self, model, meta, scaler, max_windows: int) -> int:
        frame = build_features(self._bars())
        from .features import window_end_indices
        ends = window_end_indices(frame, meta["window"])[-max_windows:]
        todo = np.array([e for e in ends if not self.store.has_analysis(self.market, int(frame.minutes[e]),
                                                                         meta["version"])], dtype=np.int64)
        if len(todo):
            with self._analysis_lock:
                self._score_windows(frame, todo, model, meta, scaler, "backfill", explain_all=True)
        LOG.info("Backfilled %d windows", len(todo))
        return int(len(todo))

    @staticmethod
    def _public(rec: dict) -> dict:
        return {k: rec[k] for k in ("id", "market", "minute", "detected_at", "score", "threshold", "is_anomaly",
                                    "model_version", "close", "features", "explanation", "source")}

    # ------------------------------------------------------------- views
    def model_info(self) -> dict:
        with self._lock:
            meta, report, error = self.meta, self.report, self.model_error
        if meta is None:
            return {"loaded": False, "error": error}
        ev = (report or {}).get("evaluation", {})
        return {"loaded": True, "version": meta["version"], "created_at": meta.get("created_at"),
                "market": meta.get("market"),
                "architecture": {"type": "LSTM AutoEncoder", "encoder": f"LSTM({meta['hidden']})",
                                 "decoder": f"LSTM({meta['hidden']})", "output": f"Linear({len(meta['features'])})",
                                 "n_params": meta.get("n_params"), "window": meta["window"],
                                 "score_steps": meta.get("score_steps")},
                "features": [{"name": f, "label": FEATURE_LABELS.get(f, f)} for f in meta["features"]],
                "threshold": meta["threshold"], "threshold_method": meta.get("threshold_method"),
                "data": meta.get("data"), "training": meta.get("training"),
                "evaluation": {"note": ev.get("note"), "score_distribution": ev.get("score_distribution"),
                               "alarm_rate": ev.get("alarm_rate"), "baseline": ev.get("baseline"),
                               "synthetic_injection": ev.get("synthetic_injection")}}

    def timeline(self, minutes: int = 180) -> dict:
        bars = self._bars()
        if not bars:
            return {"points": [], "minutes": minutes, "threshold": self.meta["threshold"] if self.meta else None,
                    "model_version": self.model_version, "latest_minute": None, "stale": True,
                    "feature_labels": FEATURE_LABELS, "message": "수집된 데이터가 없습니다."}
        frame = build_features(bars)
        start_minute = int(frame.minutes[-1]) - minutes + 1
        sel = np.nonzero(frame.minutes >= start_minute)[0]
        version = self.model_version
        scores = {r["minute"]: r for r in self.store.scores_since(self.market, start_minute, version)} if version else {}
        points = []
        for i in sel:
            m = int(frame.minutes[i])
            s = scores.get(m)
            points.append({"t": minute_iso(m), "close": float(frame.close[i]), "volume": float(frame.volume[i]),
                           "value": float(frame.value[i]), "trades": int(frame.trade_count[i]),
                           "filled": bool(frame.filled[i]), "features": frame.row_features(int(i)),
                           "score": s["score"] if s else None,
                           "is_anomaly": bool(s["is_anomaly"]) if s else None,
                           "analysis_id": s["id"] if s else None})
        now_minute = int(time.time() // 60)
        return {"points": points, "minutes": minutes, "threshold": self.meta["threshold"] if self.meta else None,
                "model_version": version, "latest_minute": minute_iso(int(frame.minutes[-1])),
                "stale": now_minute - int(frame.minutes[-1]) > STALE_MINUTES,
                "feature_labels": FEATURE_LABELS}

    def status(self) -> dict:
        with self._lock:
            meta = self.meta
            buf = len(self.buffer)
            latest = self.buffer[-1].minute if self.buffer else None
        model = None
        if meta:
            model = {"version": meta["version"], "type": meta.get("model_type"), "created_at": meta.get("created_at"),
                     "window": meta["window"], "score_steps": meta.get("score_steps"), "hidden": meta.get("hidden"),
                     "n_params": meta.get("n_params"), "threshold": meta["threshold"],
                     "threshold_method": meta.get("threshold_method"), "features": meta["features"],
                     "data_label": meta.get("data", {}).get("label"),
                     "train_period": [meta.get("data", {}).get("first_minute"), meta.get("data", {}).get("last_minute")],
                     "windows": meta.get("data", {}).get("windows")}
        need = (meta["window"] + 11) if meta else None
        return {"model_loaded": meta is not None, "model": model, "model_error": self.model_error,
                "model_loaded_at": self.model_loaded_at, "realtime_analysis": self.realtime,
                "worker_alive": bool(self._worker and self._worker.is_alive()),
                "buffer_bars": buf, "required_bars": need,
                "latest_bar": minute_iso(latest) if latest is not None else None,
                "last_analysis_at": self.last_analysis_at,
                "last_analyzed_minute": minute_iso(self.last_analyzed_minute) if self.last_analyzed_minute else None,
                "last_error": self.last_error, "store": self.store.counts(self.market, meta["version"] if meta else None),
                "bootstrap": self.bootstrap_info, "backfill": self.backfill_state}

    # ------------------------------------------------------------ worker
    def _loop(self) -> None:
        last_prune = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=5)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                bars = self._bars()
                if self.model is not None and bars and bars[-1].minute != self.last_analyzed_minute:
                    self.analyze_latest("realtime")
                    self.last_error = None
            except NotReady as nr:
                self.last_error = nr.message
            except Exception as exc:
                self.last_error = f"분석 오류: {exc}"
                LOG.exception("realtime analysis failed")
            if time.time() - last_prune > 3600:
                last_prune = time.time()
                try:
                    from . import config
                    self.store.prune(config.RETENTION_DAYS)
                except Exception:
                    LOG.exception("prune failed")

    def start(self) -> None:
        if self.realtime and (self._worker is None or not self._worker.is_alive()):
            self._stop.clear()
            self._worker = threading.Thread(target=self._loop, name="ai-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker:
            self._worker.join(timeout=5)
