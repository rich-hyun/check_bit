"""Central paths/settings. Everything can be overridden with environment variables.

Runtime data (collected CSV, SQLite DB, backups) lives in DATA_DIR, which is
git-ignored so that `git reset --hard` during deployment never touches it.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    value = os.getenv(env, "").strip()
    return Path(value).expanduser().resolve() if value else default


def _flag(env: str, default: bool = False) -> bool:
    value = os.getenv(env)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DATA_DIR = _path("CHECKBIT_DATA_DIR", BASE_DIR / "data")
RAW_CSV = _path("CHECKBIT_RAW_CSV", DATA_DIR / "btc_anomaly_dataset.csv")
BACKUP_DIR = DATA_DIR / "backup"
DB_PATH = _path("CHECKBIT_DB_PATH", DATA_DIR / "checkbit.sqlite3")
STATE_PATH = DATA_DIR / "collector_state.json"
# Model artifacts: committed under models/ (trained offline) unless overridden.
MODEL_DIR = _path("CHECKBIT_MODEL_DIR", BASE_DIR / "models" / "lstm_ae")
# Legacy location used before this change (repo root, tracked by git).
LEGACY_CSV = BASE_DIR / "btc_anomaly_dataset.csv"

MARKET = os.getenv("CHECKBIT_MARKET", "KRW-BTC").strip() or "KRW-BTC"
UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"

# Admin token protects collector start/stop and model reload. Empty -> disabled.
ADMIN_TOKEN = os.getenv("CHECKBIT_ADMIN_TOKEN", "").strip()
# None = restore the last desired state saved in STATE_PATH.
AUTOSTART = os.getenv("CHECKBIT_COLLECTOR_AUTOSTART", "").strip().lower() or None
REALTIME_ANALYSIS = _flag("CHECKBIT_REALTIME_ANALYSIS", True)
# How many recent minute bars are kept in memory for realtime inference/charts.
BUFFER_MINUTES = max(120, int(os.getenv("CHECKBIT_BUFFER_MINUTES", "720")))
# Normal (non-anomalous) analysis rows older than this are pruned from SQLite.
RETENTION_DAYS = max(1, int(os.getenv("CHECKBIT_RETENTION_DAYS", "30")))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
