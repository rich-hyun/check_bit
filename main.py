import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from checkbit_ai import api as ai_api
from checkbit_ai import config
from checkbit_ai.rawstore import RawCsvStore
from checkbit_ai.service import DetectionService
from checkbit_ai.store import Store
from collector import UpbitCollector
from market_dashboard import router as dashboard_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
LOG = logging.getLogger("checkbit")

state = {}


def _save_desired(running: bool) -> None:
    try:
        tmp = config.STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "running": running,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }), encoding="utf-8")
        tmp.replace(config.STATE_PATH)
    except OSError as exc:
        LOG.warning("Could not save collector state: %s", exc)


def _should_autostart() -> bool:
    if config.AUTOSTART in ("1", "true", "yes", "on"):
        return True
    if config.AUTOSTART in ("0", "false", "no", "off"):
        return False
    # Not configured: restore the state from before the restart.
    try:
        return bool(json.loads(config.STATE_PATH.read_text(encoding="utf-8")).get("running"))
    except (OSError, ValueError):
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    # First start after the storage move: import the newest pre-deploy backup of
    # the old git-tracked CSV (made by deploy.yml), else the legacy file itself.
    backups = sorted(config.BACKUP_DIR.glob("btc_anomaly_dataset_predeploy_*.csv"), reverse=True)
    raw = RawCsvStore(config.RAW_CSV, config.BACKUP_DIR, backups + [config.LEGACY_CSV])
    for note in raw.notes:
        LOG.info("CSV storage: %s", note)
    store = Store(config.DB_PATH)
    service = DetectionService(store, raw, config.MODEL_DIR, config.MARKET,
                               config.BUFFER_MINUTES, config.REALTIME_ANALYSIS)
    service.load_model()
    service.bootstrap()
    collector = UpbitCollector(raw, config.MARKET, on_trade=service.on_trade)
    state.update(raw=raw, store=store, service=service, collector=collector)
    ai_api.bind(service)
    service.start()
    # Score windows restored from the CSV in the background (does not block startup).
    def _backfill():
        try:
            service.backfill()
        except Exception as exc:  # e.g. shutdown while running
            LOG.warning("Backfill aborted: %s", exc)

    backfill_thread = threading.Thread(target=_backfill, name="ai-backfill", daemon=True)
    backfill_thread.start()
    if _should_autostart():
        collector.start()
    yield
    collector.stop()
    service.stop()
    backfill_thread.join(timeout=10)
    store.close()
    ai_api.bind(None)


app = FastAPI(title="CheckBit API", lifespan=lifespan)
app.include_router(dashboard_router)
app.include_router(ai_api.router)


# 프론트 연결 허용

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():

    return {
        "message":
        "Financial Data Collector API"
    }


@app.post("/collector/start")
def start(x_admin_token: str | None = Header(None)):
    ai_api.require_admin(x_admin_token)
    started = state["collector"].start()
    _save_desired(True)
    return {
        "status": "running",
        "already_running": not started
    }


@app.post("/collector/stop")
def stop(x_admin_token: str | None = Header(None)):
    ai_api.require_admin(x_admin_token)
    ended = state["collector"].stop()
    _save_desired(False)
    return {
        "status": "stopped",
        "connection_closed": ended
    }


@app.get("/status")
def status():
    # `running`, `market`, `rows` keep the original response shape.
    return state["collector"].status()


@app.get("/download")
def download():
    raw = state["raw"]
    return StreamingResponse(
        raw.iter_snapshot(),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            'attachment; filename="btc_anomaly_dataset.csv"'
        }
    )
