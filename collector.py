import websocket
import json
import csv
import os
from datetime import datetime


FILE_NAME = "btc_anomaly_dataset.csv"


if not os.path.exists(FILE_NAME):

    with open(FILE_NAME, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "market",
            "trade_price",
            "trade_volume",
            "ask_bid",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "total_bid_size",
            "total_ask_size"
        ])


latest_orderbook = {
    "bid_price":0,
    "ask_price":0,
    "bid_size":0,
    "ask_size":0,
    "total_bid_size":0,
    "total_ask_size":0
}



def save_trade(data):

    with open(
        FILE_NAME,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.writer(f)


        writer.writerow([

            datetime.now(),

            data["code"],

            data["trade_price"],

            data["trade_volume"],

            data["ask_bid"],


            latest_orderbook["bid_price"],

            latest_orderbook["ask_price"],

            latest_orderbook["bid_size"],

            latest_orderbook["ask_size"],

            latest_orderbook["total_bid_size"],

            latest_orderbook["total_ask_size"]

        ])




def save_orderbook(data):

    global latest_orderbook


    unit = data["orderbook_units"][0]


    latest_orderbook = {

        "bid_price":
            unit["bid_price"],

        "ask_price":
            unit["ask_price"],

        "bid_size":
            unit["bid_size"],

        "ask_size":
            unit["ask_size"],

        "total_bid_size":
            data["total_bid_size"],

        "total_ask_size":
            data["total_ask_size"]

    }





def on_message(ws,message):

    data=json.loads(message)


    if data["type"]=="trade":

        save_trade(data)


        print(
            "TRADE",
            data["trade_price"]
        )


    elif data["type"]=="orderbook":

        save_orderbook(data)





def on_open(ws):


    subscribe=[

        {
            "ticket":"btc_dataset"
        },


        {
            "type":"trade",
            "codes":[
                "KRW-BTC"
            ]
        },


        {
            "type":"orderbook",
            "codes":[
                "KRW-BTC"
            ]
        }

    ]


    ws.send(
        json.dumps(subscribe)
    )





def start_collector():


    ws = websocket.WebSocketApp(

        "wss://api.upbit.com/websocket/v1",

        on_message=on_message

    )


    ws.on_open = on_open


    ws.run_forever()