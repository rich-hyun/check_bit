"""Local development server that mimics the production Nginx layout:

    /        -> index.html, app.js, style.css (static)
    /api/... -> FastAPI app from main.py (prefix stripped, like Nginx)

Usage:  python dev_server.py  (then open http://127.0.0.1:8080)
Production keeps using `uvicorn main:app` behind Nginx; this file is not deployed as a service.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

import main

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app):
    async with main.lifespan(main.app):  # mounted apps do not run their own lifespan
        yield


dev = FastAPI(lifespan=lifespan)
dev.mount("/api", main.app)


@dev.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@dev.get("/{name}")
def static(name: str):
    if name not in {"app.js", "style.css"}:
        return FileResponse(ROOT / "index.html")
    return FileResponse(ROOT / name)


if __name__ == "__main__":
    uvicorn.run(dev, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
