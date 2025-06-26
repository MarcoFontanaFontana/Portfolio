"""
simulate_trader.py
────────────────────────────────────────────────────────────────────────────
A tiny Binance paper-trader that consumes the real-time CSV produced by
get_live_data(), uses a previously trained GRU (gv.NEURAL_NETWORK) and
prints PnL after every round-trip.

Requirements
------------
pip install pandas numpy joblib tensorflow
"""

import os
import gc
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ───── Your project helpers ──────────────────────────────────────────────
from RALLY_data_live import get_live_data
from RALLY_global_variables import global_variables as gv

# ───── Constants & hyper-params ─────────────────────────────────────────
CSV_PATH      = gv.LIVE_DATA        # produced by get_live_data()
MODEL_PATH    = gv.NEURAL_NETWORK   # trained GRU (.keras / .h5)
SCALER_X_PATH = "scaler_X.save"
SCALER_y_PATH = "scaler_y.save"

STAKE_FRAC    = gv.TRADE_FRACTION   # % of current wallet to risk each trade
START_USD     = gv.STARTING_BALANCE # initial cash

# ───── Utilities ────────────────────────────────────────────────────────
COLS = ["timestamp",
        "openPrice", "highPrice", "lowPrice", "lastPrice", "volume"]

def _tail_csv(path: str, n: int) -> pd.DataFrame:
    """
    Read only the last *n* rows of a CSV with pandas.  Cheap enough for
    1-second bars; switch to a file-pointer loop if you monitor many pairs.
    """
    return (
        pd.read_csv(path, names=COLS, header=0, dtype=str)
          .tail(n)
          .reset_index(drop=True)
    )

def _start_live_thread():
    """
    Fire-and-forget background thread that keeps the CSV up-to-date.
    """
    threading.Thread(
        target=get_live_data,
        kwargs=dict(symbol=gv.CRYPTO,
                    interval=gv.FREQUENCY_DATA,
                    csv_path=gv.LIVE_DATA),
        daemon=True
    ).start()
    print("📡  Live feed thread launched")

# ───── Main simulator ───────────────────────────────────────────────────
def simulate_trader():
    _start_live_thread()

    # 1) Load model and scalers
    print("⏳  Loading model & scalers …")
    model     = load_model(MODEL_PATH, compile=False)
    scaler_X  = joblib.load(SCALER_X_PATH)
    scaler_y  = joblib.load(SCALER_y_PATH)

    # 2) Wait for enough data
    print(f"⏳  Waiting for {gv.WINDOW_SIZE} bars in {CSV_PATH} …")
    while True:
        try:
            init_df = _tail_csv(CSV_PATH, gv.WINDOW_SIZE)
            if len(init_df) == gv.WINDOW_SIZE:
                break
        except Exception:
            pass
        time.sleep(0.25)
    print("✅  Warm-up complete")

    hist           = deque(init_df.to_dict("records"),
                           maxlen=gv.WINDOW_SIZE)
    wallet_usd     = START_USD
    in_position    = False
    entry_price    = None
    pred_price     = None
    trade_type     = None           # 'long' | 'short'
    horizon_cnt    = 0              # seconds held

    while True:
        # -----------------------------------------------------------------
        # 1) Pull the *next* bar (blocks until a fresh timestamp appears)
        # -----------------------------------------------------------------
        try:
            latest = _tail_csv(CSV_PATH, 1).iloc[0]
        except Exception:
            time.sleep(0.10)
            continue

        if latest["timestamp"] == hist[-1]["timestamp"]:
            time.sleep(0.10)        # bar not closed yet
            continue

        hist.append(latest)

        # -----------------------------------------------------------------
        # 2) If in position, count down until HORIZON seconds have passed
        # -----------------------------------------------------------------
        if in_position:
            horizon_cnt += 1

            if horizon_cnt >= gv.HORIZON:
                close_price = float(latest["lastPrice"])
                if trade_type == "long":
                    pnl = (close_price - entry_price) * stake_btc
                else:  # short
                    pnl = (entry_price - close_price) * stake_btc
                wallet_usd += pnl

                arrow_pred = "↑" if pred_price > entry_price else "↓"
                arrow_real = "↑" if close_price > entry_price else "↓"

                print(f"[{latest['timestamp']}] CLOSE  {trade_type.upper():5s} | "
                      f"Entry {entry_price:,.2f} | "
                      f"Pred {pred_price:,.2f}{arrow_pred} | "
                      f"Exit {close_price:,.2f}{arrow_real} | "
                      f"PnL ${pnl:,.2f} | "
                      f"Portfolio ${wallet_usd:,.2f}")

                # cool-down period
                in_position = False
                time.sleep(gv.WAIT_TIME)
                continue

        # -----------------------------------------------------------------
        # 3) If flat, open a new trade using the last WINDOW_SIZE rows
        # -----------------------------------------------------------------
        if not in_position:
            # Build model input
            df_hist = pd.DataFrame(hist)
            X = df_hist[["openPrice","highPrice","lowPrice","lastPrice","volume"]].astype(float).values
            X = X.reshape(1, gv.WINDOW_SIZE, 5)
            Xs = scaler_X.transform(X.reshape(-1,5)).reshape(X.shape)

            # Predict
            pred_scaled = model.predict(Xs, verbose=0)
            pred_price  = scaler_y.inverse_transform(pred_scaled)[0,0]

            entry_price = float(latest["lastPrice"])
            trade_type  = "long" if pred_price > entry_price else "short"

            # Size position: fixed fraction of wallet, no leverage
            stake_usd = STAKE_FRAC * wallet_usd
            stake_btc = stake_usd / entry_price

            arrow = "↑" if trade_type=="long" else "↓"
            print(f"[{latest['timestamp']}] OPEN   {trade_type.upper():5s} | "
                  f"Entry {entry_price:,.2f} | "
                  f"Pred {pred_price:,.2f}{arrow} | "
                  f"Size ${stake_usd:,.2f} | "
                  f"Portfolio ${wallet_usd:,.2f}")

            # Activate position bookkeeping
            in_position  = True
            horizon_cnt  = 0

        # tidy-up
        gc.collect()


# ───── entry-point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        simulate_trader()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
