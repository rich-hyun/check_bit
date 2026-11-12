"""SQLite storage for analysis results (detection history) and user feedback.

SQLite is file based (no paid DB service) and safe for this single-process
FastAPI app; all access goes through one lock and short transactions.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    market        TEXT    NOT NULL,
    minute        INTEGER NOT NULL,          -- window end, epoch minute (UTC)
    detected_at   TEXT    NOT NULL,          -- ISO time of window end
    score         REAL    NOT NULL,
    threshold     REAL    NOT NULL,
    is_anomaly    INTEGER NOT NULL,
    model_version TEXT    NOT NULL,
    close         REAL,
    features      TEXT    NOT NULL,          -- JSON: raw feature values of the last minute
    explanation   TEXT,                      -- JSON: rule-based evidence (anomalies + latest)
    source        TEXT    NOT NULL,          -- realtime | manual | backfill
    created_at    TEXT    NOT NULL,
    UNIQUE (market, minute, model_version)
);
CREATE INDEX IF NOT EXISTS idx_analyses_minute ON analyses (market, minute);
CREATE INDEX IF NOT EXISTS idx_analyses_anomaly ON analyses (is_anomaly, minute);
CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id  INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    label        TEXT    NOT NULL CHECK (label IN ('normal', 'anomaly', 'hold')),
    note         TEXT,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON feedback (analysis_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------- analyses
    def save_analysis(self, rec: dict) -> int:
        """Insert (or keep existing) analysis for (market, minute, model_version)."""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO analyses (market, minute, detected_at, score, threshold, is_anomaly,"
                " model_version, close, features, explanation, source, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["market"], rec["minute"], rec["detected_at"], rec["score"], rec["threshold"],
                 int(rec["is_anomaly"]), rec["model_version"], rec.get("close"),
                 json.dumps(rec["features"], ensure_ascii=False),
                 json.dumps(rec["explanation"], ensure_ascii=False) if rec.get("explanation") else None,
                 rec["source"], _now()))
            if cur.lastrowid and cur.rowcount:
                self._db.commit()
                return int(cur.lastrowid)
            row = self._db.execute("SELECT id FROM analyses WHERE market=? AND minute=? AND model_version=?",
                                   (rec["market"], rec["minute"], rec["model_version"])).fetchone()
            return int(row["id"])

    def has_analysis(self, market: str, minute: int, model_version: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM analyses WHERE market=? AND minute=? AND model_version=?",
                                    (market, minute, model_version)).fetchone() is not None

    @staticmethod
    def _row(r: sqlite3.Row, full: bool = False) -> dict:
        d = {"id": r["id"], "market": r["market"], "minute": r["minute"], "detected_at": r["detected_at"],
             "score": r["score"], "threshold": r["threshold"], "is_anomaly": bool(r["is_anomaly"]),
             "model_version": r["model_version"], "close": r["close"], "source": r["source"],
             "created_at": r["created_at"], "features": json.loads(r["features"])}
        if "fb_label" in r.keys():
            d["user_label"] = r["fb_label"]
        exp = json.loads(r["explanation"]) if r["explanation"] else None
        if full:
            d["explanation"] = exp
        else:
            d["summary"] = exp.get("summary") if exp else None
            d["top_features"] = [{"feature": t["feature"], "label": t["label"],
                                  "contribution_pct": t["contribution_pct"]} for t in exp.get("top_features", [])] if exp else []
        return d

    _LATEST_FB = ("(SELECT label FROM feedback f WHERE f.analysis_id = a.id ORDER BY f.id DESC LIMIT 1) AS fb_label")

    def list_analyses(self, market: str, *, anomalies_only: bool = False, limit: int = 50,
                      since_minute: int | None = None, until_minute: int | None = None,
                      model_version: str | None = None) -> list[dict]:
        q = f"SELECT a.*, {self._LATEST_FB} FROM analyses a WHERE market=?"
        args: list = [market]
        if anomalies_only:
            q += " AND is_anomaly=1"
        if since_minute is not None:
            q += " AND minute>=?"
            args.append(since_minute)
        if until_minute is not None:
            q += " AND minute<=?"
            args.append(until_minute)
        if model_version:
            q += " AND model_version=?"
            args.append(model_version)
        q += " ORDER BY minute DESC, id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [self._row(r) for r in self._db.execute(q, args).fetchall()]

    def get_analysis(self, analysis_id: int) -> dict | None:
        with self._lock:
            r = self._db.execute(f"SELECT a.*, {self._LATEST_FB} FROM analyses a WHERE id=?",
                                 (analysis_id,)).fetchone()
            if r is None:
                return None
            d = self._row(r, full=True)
            d["feedback"] = [dict(x) for x in self._db.execute(
                "SELECT id, label, note, created_at FROM feedback WHERE analysis_id=? ORDER BY id DESC",
                (analysis_id,)).fetchall()]
            return d

    def latest(self, market: str, model_version: str | None = None) -> dict | None:
        q = f"SELECT a.*, {self._LATEST_FB} FROM analyses a WHERE market=?"
        args: list = [market]
        if model_version:
            q += " AND model_version=?"
            args.append(model_version)
        with self._lock:
            r = self._db.execute(q + " ORDER BY minute DESC, id DESC LIMIT 1", args).fetchone()
            return self._row(r, full=True) if r else None

    def scores_since(self, market: str, since_minute: int, model_version: str | None) -> list[dict]:
        q = "SELECT id, minute, score, threshold, is_anomaly FROM analyses WHERE market=? AND minute>=?"
        args: list = [market, since_minute]
        if model_version:
            q += " AND model_version=?"
            args.append(model_version)
        with self._lock:
            return [dict(r) for r in self._db.execute(q + " ORDER BY minute", args).fetchall()]

    def counts(self, market: str, model_version: str | None = None) -> dict:
        q = "SELECT COUNT(*) n, COALESCE(SUM(is_anomaly),0) a, MAX(created_at) last FROM analyses WHERE market=?"
        args: list = [market]
        if model_version:
            q += " AND model_version=?"
            args.append(model_version)
        with self._lock:
            r = self._db.execute(q, args).fetchone()
            return {"analyses": r["n"], "anomalies": r["a"], "last_saved_at": r["last"]}

    def prune(self, retention_days: int) -> int:
        """Delete old *normal* rows without feedback; anomalies are kept."""
        cutoff = int(time.time() // 60) - retention_days * 24 * 60
        with self._lock:
            cur = self._db.execute("DELETE FROM analyses WHERE is_anomaly=0 AND minute<? AND id NOT IN "
                                   "(SELECT analysis_id FROM feedback)", (cutoff,))
            self._db.commit()
            return cur.rowcount

    # ------------------------------------------------------------- feedback
    def add_feedback(self, analysis_id: int, label: str, note: str | None) -> dict | None:
        with self._lock:
            if self._db.execute("SELECT 1 FROM analyses WHERE id=?", (analysis_id,)).fetchone() is None:
                return None
            cur = self._db.execute("INSERT INTO feedback (analysis_id, label, note, created_at) VALUES (?,?,?,?)",
                                   (analysis_id, label, note, _now()))
            self._db.commit()
            return {"id": cur.lastrowid, "analysis_id": analysis_id, "label": label, "note": note}

    def feedback_summary(self, market: str) -> dict:
        """Compare AI decision with the latest user label per analysis."""
        q = ("SELECT a.is_anomaly ai, f.label user FROM analyses a JOIN feedback f ON f.analysis_id=a.id "
             "WHERE a.market=? AND f.id = (SELECT MAX(id) FROM feedback WHERE analysis_id=a.id)")
        with self._lock:
            rows = self._db.execute(q, (market,)).fetchall()
        m = {f"ai_{'anomaly' if ai else 'normal'}__user_{u}": 0 for ai in (0, 1) for u in ("normal", "anomaly", "hold")}
        for r in rows:
            m[f"ai_{'anomaly' if r['ai'] else 'normal'}__user_{r['user']}"] += 1
        decided = [r for r in rows if r["user"] != "hold"]
        agree = sum(1 for r in decided if (r["user"] == "anomaly") == bool(r["ai"]))
        ai_anom = [r for r in decided if r["ai"]]
        return {"reviewed": len(rows), "decided": len(decided), "hold": len(rows) - len(decided), "matrix": m,
                "agreement_rate": (agree / len(decided)) if decided else None,
                "user_confirmed_precision": (sum(1 for r in ai_anom if r["user"] == "anomaly") / len(ai_anom))
                if ai_anom else None,
                "note": "사용자가 검토한 항목만 집계한 값이며, 전체 탐지 성능을 대표하지 않습니다."}
