import requests
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime, timedelta
import os
import pytz
from RALLY_global_variables import global_variables as gv


def get_historical_data(symbol=gv.CRYPTO, interval=gv.FREQUENCY_DATA):
    url = "https://api.binance.com/api/v3/klines"
    # Calculate start and end timestamps using global start/end tuples
    start_time = int(datetime(*gv.START_DATE).timestamp() * 1000)
    end_time = int(datetime(*gv.END_DATE).timestamp() * 1000)
        
    # Prepare CSV file: if it doesn't exist, create it with headers.
    if not os.path.exists(gv.HISTORICAL_DATA):
        print(f"📄 {gv.HISTORICAL_DATA} not found. Creating new file with historical data...")
        header = ["timestamp", "openPrice", "highPrice", "lowPrice", "lastPrice", "volume"]
        with open(gv.HISTORICAL_DATA, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    rome_tz = pytz.timezone("Europe/Rome")
    #total_records = 0

    # Loop and fetch data in chunks of 1000 candles
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Max allowed per request
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data or isinstance(data, dict):  # API rate limit or error object
                print("⚠️ No more data received. Binance might be rate-limiting requests.")
                break

            # Convert chunk to DataFrame with proper column names
            df_chunk = pd.DataFrame(data, columns=[
                "timestamp", "openPrice", "highPrice", "lowPrice", "lastPrice",
                "volume", "_", "_", "_", "_", "_", "_"
            ])
            df_chunk = df_chunk[["timestamp", "openPrice", "highPrice", "lowPrice", "lastPrice", "volume"]]
            # Convert timestamp from ms and adjust to Rome local time
            df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="ms")
            df_chunk["timestamp"] = df_chunk["timestamp"].dt.tz_localize("UTC").dt.tz_convert(rome_tz)
            df_chunk["timestamp"] = df_chunk["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Append chunk rows immediately to the CSV
            with open(gv.HISTORICAL_DATA, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(df_chunk.values)
            
            #records_count = len(df_chunk)
            #total_records += records_count
            #print(f"✅ Retrieved and appended {records_count} records... (Total: {total_records})")
            
            # Update start_time for next iteration:
            # We convert the last row's timestamp string back to UTC timestamp.
            last_ts_str = df_chunk.iloc[-1]["timestamp"]
            last_dt = datetime.strptime(last_ts_str, '%Y-%m-%d %H:%M:%S')
            # Localize to Rome time, then convert to UTC
            last_dt_local = rome_tz.localize(last_dt)
            last_dt_utc = last_dt_local.astimezone(pytz.utc)
            start_time = int(last_dt_utc.timestamp() * 1000) + 1

            # Brief pause
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Binance API request failed: {e}")
            break

    #print(f"✅ Historical prices appended: {total_records} records")
    #return total_records

if __name__ == "__main__":
    start = time.time()
    get_historical_data()
    end = time.time()
    elapsed = (end - start) / 60
    print(f"Execution time: {elapsed:.2f} minutes")
