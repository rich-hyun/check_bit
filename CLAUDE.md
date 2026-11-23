# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CheckBit monitors KRW-BTC on Upbit. It is a FastAPI backend with a static dashboard and has three parts:

1. **Collector** (`collector.py`): reads Upbit WebSocket trades and orderbook.
   - It writes every trade to a CSV (`data/btc_anomaly_dataset.csv`).
   - It also writes the raw messages to SQLite (`data/raw.sqlite3`): all trades, 15-level orderbook snapshots, and connection sessions.
2. **AI detection** (`checkbit_ai/`): builds 1-minute bars and 11 features from the CSV.
   - A NumPy LSTM AutoEncoder scores each window.
   - Analyses and user feedback are stored in `data/checkbit.sqlite3`.
3. **Dashboard** (`index.html`, `app.js`, `style.css`): live collector status, the AI section, and the older REST-based price chart with median/MAD bands (`market_dashboard.py`).

`README.md` (Korean) is the detailed reference for the API table, environment variables, model card, and training procedure. Read it before changing behaviour. Comments and user-facing strings are mostly Korean.

The project goal is in `docs/PROJECT_CONTEXT.md`: anomaly candidates are triggered by detectors, then an **agent** gathers evidence and gives a verdict, and a human makes the final call. The LSTM AE is one detector in that plan. The agent does not exist yet.

## Commands

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest
pytest -q                             # all tests use synthetic data; ~15s
pytest -q tests/test_rawdb.py         # one file
python dev_server.py                  # http://127.0.0.1:8080, mimics Nginx (/ static, /api -> FastAPI)
python run.py                         # API only, 127.0.0.1:8000
python -m checkbit_ai.train --csv <collected.csv> --out models/lstm_ae   # offline training
```

Admin endpoints (`/collector/start|stop`, `/ai/model/reload`) need the header `X-Admin-Token` to match `CHECKBIT_ADMIN_TOKEN`. If the token is unset they return 403.

## Architecture

- **Wiring** (`main.py` lifespan):
  - Objects: `RawCsvStore` (CSV), `Store` (analyses DB), `DetectionService`, `RawDbWriter` (raw DB), and `UpbitCollector(raw, on_trade=service.on_trade, rawdb=rawdb)`.
  - Collector autostart: `CHECKBIT_COLLECTOR_AUTOSTART`. If unset, the last desired state is restored from `data/collector_state.json`.
- **Two SQLite files, two env vars. Do not mix them up.**
  - `CHECKBIT_DB_PATH` → `checkbit.sqlite3`: analyses and feedback (`checkbit_ai/store.py`).
  - `CHECKBIT_RAW_DB_PATH` → `raw.sqlite3`: raw market data (`checkbit_ai/rawdb.py`).
- **Collector** (`collector.py`, `UpbitCollector`): one thread with reconnect backoff, plus a watchdog that forces a reconnect after 45s without messages.
  - `start()` is idempotent, and `stop()` closes the socket.
  - CSV dedup is an in-memory set of the last 50k `sequential_id`s, seeded from the CSV tail on startup.
  - The CSV's orderbook columns are the level-1 book *as last received* when the trade arrived. There is no orderbook timestamp.
- **Raw DB** (`checkbit_ai/rawdb.py`, `RawDbWriter`):
  - The collector hands every trade and orderbook message to `submit()` *before* its own CSV dedup.
  - Writes go through a queue to a single writer thread that commits in batches (500 items or 1s).
  - `trades` has PK `(market, sequential_id)`, so dedup survives restarts.
  - Orderbook snapshots are sampled by Upbit timestamp (`CHECKBIT_ORDERBOOK_INTERVAL_MS`, default 1000, 0 = keep all). A duplicate snapshot is detected with `changes()` and its levels are skipped.
  - `collector_sessions`: the collector's `_on_open` opens a row, and `_close_session()` closes it with reason `stop`, `stale: …`, `error: …`, or `closed`.
  - Session events are never dropped. Rows left open by a killed process become `unclean_shutdown` on the next `start()`, but only after the lock is taken.
  - Messages missing required fields (the old fake server sends orderbooks with no `timestamp`) are counted in `skipped_invalid`, not raised.
  - `RawDbWriter.start()` takes a non-blocking `fcntl.flock` on `raw.sqlite3.lock`. If the lock is held, `UpbitCollector.start()` raises `CollectorLockedError`. `/collector/start` turns that into a 409, and autostart logs it and keeps serving.
- **Features** (`checkbit_ai/features.py`): trades become `MinuteBar`s, then `build_features`.
  - Gaps of up to 5 missing minutes are filled as zero-volume, flat-price bars. Longer gaps start a new segment, and windows never cross segments.
  - This does **not** consult `collector_sessions` yet, so a short outage looks like "no trading".
  - The scaler is fitted on the training split only and saved with the model.
  - Feature order in `FEATURES` is part of the saved model. Do not reorder it without retraining.
- **Realtime service** (`checkbit_ai/service.py`):
  - On startup it loads the model and bootstraps bars from the last 16MB of the CSV.
  - It aggregates live trades via `on_trade` and analyzes every minute (3s grace).
  - It backfills unscored windows in a background thread.
  - With no model it reports "not ready" and never fabricates results.
- **Explanations** (`checkbit_ai/explain.py`): rule-based sentences built only from computed numbers, using training-period percentiles as "normal".

## Gotchas

- `checkbit_ai/config.py` paths are module constants that tests `monkeypatch`. New path settings should be functions (like `raw_db_path()`) or be read inside the lifespan, or tests will write into the real `data/`.
- Tests use only synthetic data (`checkbit_ai/synthetic.py`) and local fake WebSocket servers. `conftest.retry_winsock` exists for a Windows-only socket flake.
- Times: Upbit market times are UTC epoch ms. Session and `received_at` times are server ms. The UI shows KST.
- Upbit resends a `SNAPSHOT` trade and orderbook on every (re)connect. They can carry a timestamp slightly *before* `connected_at`.

## Deployment

A push to `main` triggers `.github/workflows/deploy.yml`, which runs on the Lightsail host in `/home/ubuntu/check_bit`:

1. Writes a systemd drop-in (`CHECKBIT_COLLECTOR_AUTOSTART=1`, and `CHECKBIT_ADMIN_TOKEN` from GitHub secrets).
2. Backs up a root `btc_anomaly_dataset.csv` if there is one, then runs `git reset --hard origin/main` and `pip install`.
3. Aborts if Python < 3.10 or `import main` fails, so the running version keeps serving.
4. Copies `index.html`, `style.css`, and `app.js` to `/var/www/checkbit/`, then restarts `checkbit`, reloads nginx, and smoke tests `/status` and `https://checkbit.site`.

- Nginx proxies `/api/*` to FastAPI with the prefix stripped, so the frontend calls `/api/...`.
- Everything in `data/` is git-ignored and survives deploys.
- If you rename the frontend files, update the workflow's `cp` steps too. Keep `/status` working.
- Work on a branch and merge by PR. Never push to `main` directly.

@docs/PROJECT_CONTEXT.md
