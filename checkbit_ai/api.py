"""FastAPI router for AI anomaly detection (mounted at /ai, i.e. /api/ai behind Nginx).

Read endpoints are public. Model reload requires the admin token. Training is
not exposed over HTTP at all (offline CLI only).
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import config
from .service import DetectionService, NotReady

router = APIRouter(prefix="/ai", tags=["AI"])
_service: DetectionService | None = None


def bind(service: DetectionService | None) -> None:
    global _service
    _service = service


def svc() -> DetectionService:
    if _service is None:
        raise HTTPException(503, "AI 서비스가 초기화되지 않았습니다.")
    return _service


def require_admin(token: str | None) -> None:
    expected = config.ADMIN_TOKEN
    if not expected:
        raise HTTPException(403, "서버에 CHECKBIT_ADMIN_TOKEN이 설정되지 않아 관리 기능이 비활성화되어 있습니다.")
    if not token or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(401, "관리자 토큰이 올바르지 않습니다.")


class _RateLimit:
    def __init__(self, limit: int, per_seconds: float) -> None:
        self.limit, self.per = limit, per_seconds
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.per:
                q.popleft()
            if len(q) >= self.limit:
                raise HTTPException(429, "요청이 너무 많습니다. 잠시 후 다시 시도하세요.")
            q.append(now)
            if len(self.hits) > 5000:
                self.hits.clear()


_analyze_limit = _RateLimit(20, 60)
_feedback_limit = _RateLimit(30, 60)


def client_key(request: Request) -> str:
    # Behind Nginx the peer is 127.0.0.1; use the forwarded address when present.
    return (request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown"))


@router.get("/status")
def ai_status():
    return svc().status()


@router.get("/model")
def model_info():
    """Model architecture, training data summary and evaluation report."""
    return svc().model_info()


@router.post("/analyze")
def analyze(request: Request):
    """Run the model on the most recent complete window now (results are cached per minute)."""
    _analyze_limit.check(client_key(request))
    try:
        return svc().analyze_latest("manual")
    except NotReady as nr:
        return nr.as_dict()


@router.get("/analysis/latest")
def latest_analysis():
    s = svc()
    if s.model is None:
        return {"status": "model_not_ready", "message": "학습된 이상탐지 모델이 준비되지 않았습니다.",
                "detail": s.model_error}
    rec = s.store.latest(s.market, s.model_version)
    if rec is None:
        r = s.readiness()
        return {"status": "warming_up", "message": f"실시간 분석 준비 중입니다 ({r['ready_bars']}/{r['required_bars']}분).", **r}
    stale = int(time.time() // 60) - rec["minute"] > 3
    return {"status": "ok", **rec, "stale": stale,
            "message": "최신 수집 데이터가 아닙니다. 데이터 수집 상태를 확인하세요." if stale else None}


@router.get("/timeline")
def timeline(minutes: int = Query(180, ge=30, le=720)):
    return svc().timeline(minutes)


@router.get("/detections")
def detections(limit: int = Query(30, ge=1, le=500), anomalies_only: bool = True,
               since: int | None = Query(None, description="epoch minute (UTC)"),
               until: int | None = Query(None, description="epoch minute (UTC)"),
               all_models: bool = False):
    s = svc()
    items = s.store.list_analyses(s.market, anomalies_only=anomalies_only, limit=limit, since_minute=since,
                                  until_minute=until, model_version=None if all_models else s.model_version)
    return {"items": items, "count": len(items), "model_version": s.model_version}


@router.get("/detections/{analysis_id}")
def detection_detail(analysis_id: int):
    rec = svc().store.get_analysis(analysis_id)
    if rec is None:
        raise HTTPException(404, "해당 분석 결과가 없습니다.")
    return rec


class FeedbackIn(BaseModel):
    label: Literal["normal", "anomaly", "hold"]
    note: str | None = Field(None, max_length=500)


@router.post("/detections/{analysis_id}/feedback")
def add_feedback(analysis_id: int, body: FeedbackIn, request: Request):
    _feedback_limit.check(client_key(request))
    note = body.note.strip() if body.note else None
    saved = svc().store.add_feedback(analysis_id, body.label, note or None)
    if saved is None:
        raise HTTPException(404, "해당 분석 결과가 없습니다.")
    return saved


@router.get("/feedback/summary")
def feedback_summary():
    s = svc()
    return s.store.feedback_summary(s.market)


@router.post("/model/reload")
def reload_model(x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    s = svc()
    ok = s.load_model()
    filled = s.backfill() if ok else 0
    return {"loaded": ok, "version": s.model_version, "error": s.model_error, "backfilled": filled}
