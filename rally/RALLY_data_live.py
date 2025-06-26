################################ DATA DOWNLOADER ################################

import requests
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime, timedelta
import os
import pytz
from RALLY_global_variables import global_variables as gv


# ------------------- Live data from Binance (1-second bars) ------------------- #
def get_live_data(
        symbol: str = gv.CRYPTO,
        interval: str = gv.FREQUENCY_DATA,                     # 1-second candlesticks
        csv_path: str = gv.LIVE_DATA
):
    """
    Continuously pulls the latest CLOSED 1-second kline from Binance
    and appends it to `csv_path` (timestamp, openPrice, highPrice,
    lowPrice, lastPrice, volume).  Duplicates are avoided by tracking
    the last saved timestamp.
    """
    rome_tz = pytz.timezone("Europe/Rome")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 1}

    # Create the CSV with header if it doesn’t exist yet
    if not os.path.isfile(csv_path):
        header = ["timestamp", "openPrice",
                  "highPrice", "lowPrice", "lastPrice", "volume"]
        pd.DataFrame(columns=header).to_csv(csv_path, index=False)

    last_saved_ts = None   # remember the most-recent bar we wrote

    print(f"📡 Live 1-second feed started for {symbol} …")
    while True:
        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            kline = r.json()[0]                # single 1-second bar

            # ┌──────── Binance kline payload indices ─────────┐
            # │ 0 openTime  1 open  2 high  3 low  4 close     │
            # │ 5 volume    6 closeTime … (we ignore the rest) │
            # └─────────────────────────────────────────────────┘
            ts_utc = datetime.utcfromtimestamp(kline[0] / 1000).replace(tzinfo=pytz.utc)
            ts_local = ts_utc.astimezone(rome_tz).strftime('%Y-%m-%d %H:%M:%S')

            # skip if we already stored this bar
            if ts_local == last_saved_ts:
                time.sleep(0.15)          # wait a bit, bar not closed yet
                continue

            row = [ts_local,
                   float(kline[1]),  # openPrice
                   float(kline[2]),  # highPrice
                   float(kline[3]),  # lowPrice
                   float(kline[4]),  # lastPrice (close)
                   float(kline[5])]  # volume

            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(row)

            last_saved_ts = ts_local
            # Optional: print every N seconds to avoid console spam
            # if int(ts_local[-2:]) % 10 == 0:
            #     print(f"{ts_local} appended")

            # Sleep until the next second boundary
            now = datetime.utcnow()
            time.sleep(max(0, 1 - now.microsecond / 1_000_000))

        except Exception as e:
            # Network hiccup / rate limit — back off briefly
            print(f"⚠️  Live fetch error: {e}")
            time.sleep(1)

###################################################################################################################

if __name__ == "__main__":
    start_time = time.time()

    get_live_data()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Execution time: {elapsed_time:.2f} minutes")
