#IND
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np



def get_cmma_signal( 
        data_btc: pd.DataFrame, 
        data_alt: pd.DataFrame, 
        lookback: int, 
        atr_lookback: int, 
        entry_threshold: float, 
        exit_threshold: float, 
        ema: bool = False,
        use_zlema: bool = False,
        use_hma: bool = False,
        src: bool = False,
        ema_filter_period: int = 600,
    ) -> tuple[pd.Series, pd.Series]:
    """
    calcule les cmma pour btc et l'altcoin, puis renvoie :
      - intermarket_diff : Series (alt_cmma - btc_cmma)
      - signal           : Series de mêmes index, valeurs {1, 0}
    """

    btc_cmma = cmma(data_btc, lookback, atr_lookback, ema=ema, use_zlema=use_zlema, use_hma=use_hma, src=src,)
    alt_cmma = cmma(data_alt, lookback, atr_lookback, ema=ema, use_zlema=use_zlema, use_hma=use_hma, src=src,)

    diff = (alt_cmma - btc_cmma).iloc[-1]
    ema_filter = data_alt['close'].ewm(span=ema_filter_period, adjust=False).mean().iloc[-1]
    last_close = data_alt['close'].iloc[-1]

    if diff > entry_threshold and last_close > ema_filter:
        signal = 1
    elif diff < exit_threshold:
        signal = -1
    else:
        signal = 0  # neutre

    return diff, signal
    

def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168, ema=False, use_zlema=False,use_hma=False, src=False):
    atr = atr_f(ohlc, atr_lookback)

    price_series = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3 if src else ohlc['close']
    if use_hma:
        ma = hma(price_series, lookback)
    elif use_zlema:
        ma = zlema(price_series, lookback)
    elif ema:
        ma = price_series.ewm(span=lookback, adjust=True).mean()
    else:
        ma = price_series.rolling(lookback).mean()

    ind = (ohlc['close'] - ma) / (atr * lookback ** 0.5)
    return ind


def atr_f(ohlc: pd.DataFrame, lookback: int) -> pd.Series:
    high = ohlc['high'].astype(float)
    low = ohlc['low'].astype(float)
    close = ohlc['close'].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=lookback, min_periods=lookback).mean()

    return atr

def hma(series: pd.Series, length: int) -> pd.Series:
    import numpy as np
    import pandas as pd
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    wma_half = series.rolling(window=half_length).mean()
    wma_full = series.rolling(window=length).mean()
    diff = 2 * wma_half - wma_full

    hma = diff.rolling(window=sqrt_length).mean()
    return hma


def zlema(series: pd.Series, period: int) -> pd.Series:
    lag = (period - 1) // 2
    adjusted = series + (series - series.shift(lag))
    return adjusted.ewm(span=period, adjust=False).mean()
