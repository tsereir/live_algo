import ccxt

import pandas as pd
import numpy as  np
from collections import deque


def get_last_closed_ohlcv(exchange, pair, timeframe='1m' , only_close: bool = False):
    """
    Récupère la dernière bougie clôturée pour une paire donnée.
    """
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=2)
    # if len(ohlcv) == 1:
    #     return ohlcv[-1]
    # elif len(ohlcv) == 2:
    #     return ohlcv[-2]
    # else:
    #     raise ValueError(f"Nombre inattendu de bougies pour {pair}: {len(ohlcv)}")
    if len(ohlcv) == 1:
        raw = ohlcv[-1]
    elif len(ohlcv) == 2:
        raw = ohlcv[-2]
    else:
        raise ValueError(f"Nombre inattendu de bougies pour {pair}: {len(ohlcv)}")

    timestamp, o, h, l, c, v = raw
    date = pd.to_datetime(timestamp, unit='ms')

    if only_close:
        return {"date": date, "close": c}
    else:
        return {
            "date":   date,
            "open":   o,
            "high":   h,
            "low":    l,
            "close":  c,
            "volume": v,
        }


def fetch_latest_ohlcv_to_deque(pair, exchange, timeframe='5m', maxlen=1000, only_close: bool = False):
    """
    Used to get historical data for live trading.
    """
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=maxlen)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit='ms')
    # On retire la dernière bougie (potentiellement non finalisée)
    df = df.iloc[:-1]

    if only_close:
        records = df[["date", "close"]].to_dict("records")
    else:
        records = df[["date", "open", "high", "low", "close", "volume"]].to_dict("records")

    cache = deque(records, maxlen=maxlen)
    return cache
