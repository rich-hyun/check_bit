import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import threading
import os
import csv


from collector import start_collector



app = FastAPI()



# 프론트 연결 허용

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



collector_running = False



def run_collector():

    global collector_running

    collector_running = True

    start_collector()



@app.get("/")
def home():

    return {
        "message":
        "Financial Data Collector API"
    }



@app.post("/collector/start")
def start():

    global collector_running


    if not collector_running:

        thread = threading.Thread(
            target=run_collector,
            daemon=True
        )

        thread.start()

    return {
        "status":"running"
    }




@app.post("/collector/stop")
def stop():

    global collector_running

    collector_running=False

    return {
        "status":"stopped"
    }




@app.get("/status")
def status():

    rows=0

    if os.path.exists(
        "btc_anomaly_dataset.csv"
    ):


        with open(
            "btc_anomaly_dataset.csv",
            encoding="utf-8"
        ) as f:

            rows=sum(1 for _ in f)-1



    return {

        "running":
        collector_running,

        "market":
        "KRW-BTC",

        "rows":
        rows

    }





@app.get("/download")
def download():

    return FileResponse(

        "btc_anomaly_dataset.csv",

        filename=
        "btc_anomaly_dataset.csv"

    )