#IND
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import numpy as np



def get_cmma_signal( data_btc: pd.DataFrame, data_alt: pd.DataFrame, lookback: int, atr_lookback: int, entry_threshold: float, exit_threshold: float, ema: bool = False,
    use_zlema: bool = False,
    use_hma: bool = False,
    src: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    calcule les cmma pour btc et l'altcoin, puis renvoie :
      - intermarket_diff : Series (alt_cmma - btc_cmma)
      - signal           : Series de mêmes index, valeurs {1, 0}
    """

    btc_cmma = cmma(data_btc, lookback, atr_lookback, ema=ema, use_zlema=use_zlema, use_hma=use_hma, src=src,)
    alt_cmma = cmma(data_alt, lookback, atr_lookback, ema=ema, use_zlema=use_zlema, use_hma=use_hma, src=src,)

    diff = (alt_cmma - btc_cmma).iloc[-1]

    if diff > entry_threshold: 
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


# def cmma(ohlc: pd.DataFrame, lookback: int, atr_lookback: int = 168, ema=False, zlema=False, src=False):
#     # cmma = Close minus moving average
#     atr = atr_f(ohlc, atr_lookback)
#     price_series = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3 if src else ohlc['close']
#     if zlema:
#         ma = zlema(price_series, lookback)
#     elif ema:
#         ma = price_series.ewm(span=lookback, adjust=True).mean()
#     else:
#         ma = price_series.rolling(lookback).mean()

#     ind = (ohlc['close'] - ma) / (atr * lookback ** 0.5)
#     return ind


class Signal(ABC):
    @abstractmethod
    def generate(self, df) -> pd.Series:
        """
        Prend un dictionnaire de DataFrames (ex: {"5m": df_5min, "1h": df_1h})
        et retourne une Series booléenne de signaux alignée sur df["5m"]
        """
        pass

def compute_log_atr(data, lookback):
    # Log des prix
    log_high = np.log(data['high'])
    log_low = np.log(data['low'])
    log_close = np.log(data['close'])

    # Calcul du True Range en log
    prev_close = log_close.shift(1)
    tr1 = log_high - log_low
    tr2 = (log_high - prev_close).abs()
    tr3 = (log_low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Moyenne mobile simple du True Range = ATR
    atr = true_range.rolling(window=lookback).mean()

    return atr

class ZLEMATriple:
    def __init__(self, period_fast: int = 8, period_medium: int = 21, period_slow: int = 55, use_kalman: bool = True):
        self.period_fast = period_fast
        self.period_medium = period_medium
        self.period_slow = period_slow
        self.use_kalman = use_kalman

    def zlema(self, series: pd.Series, period: int) -> pd.Series:
        lag = int(round((period - 1) / 2))
        ema_input = series + (series - series.shift(lag))
        return ema_input.ewm(span=period, adjust=False).mean()

    def kalman_filter(self, series: pd.Series) -> pd.Series:
        value1 = pd.Series(index=series.index, dtype=float)
        value2 = pd.Series(index=series.index, dtype=float)
        value3 = pd.Series(index=series.index, dtype=float)
        tr = series.diff().abs()

        for i in range(1, len(series)):
            v1 = 0.2 * (series.iloc[i] - series.iloc[i-1]) + 0.8 * (value1.iloc[i-1] if pd.notna(value1.iloc[i-1]) else 0)
            v2 = 0.1 * tr.iloc[i] + 0.8 * (value2.iloc[i-1] if pd.notna(value2.iloc[i-1]) else tr.iloc[i])
            lambd = abs(v1 / v2) if v2 != 0 else 0.0001
            alpha = (-lambd**2 + np.sqrt(lambd**4 + 16 * lambd**2)) / 8
            v3 = alpha * series.iloc[i] + (1 - alpha) * (value3.iloc[i-1] if pd.notna(value3.iloc[i-1]) else series.iloc[i])

            value1.iloc[i] = v1
            value2.iloc[i] = v2
            value3.iloc[i] = v3

        return value3
   
   
    def generate(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)


        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        # condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])
        crossover = (df["ma1"] > df["ma2"]) & (df["ma1"].shift(1) <= df["ma2"].shift(1))
        # condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])
        condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])

        if up1:
            condition &= df["ma1"].diff() > 0
            # condition &=  np.gradient(ma1) > 0
        if up2:
            condition &= df["ma2"].diff() > 0
        if up3:
            condition &= df["ma3"].diff() > 0

        condition &= crossover

        signal = pd.Series(0, index=df.index, name="zlema_signal")
        signal[condition] = 1

        signal = signal.shift(1)

        return signal
   
    def generate_ema_signal(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
       
        # Source de prix : prix typique
        src = (df['high'] + df['low'] + df['close']) / 3

        # Filtrage optionnel par Kalman
        if self.use_kalman:
            src = self.kalman_filter(src)

        # Moyennes mobiles exponentielles
        ma1 = src.ewm(span=self.period_fast, adjust=False).mean()
        ma2 = src.ewm(span=self.period_medium, adjust=False).mean()
        ma3 = src.ewm(span=self.period_slow, adjust=False).mean()

        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        # Condition d'ordre croissant des EMAs
        condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])

        if up1:
            condition &= df["ma1"].diff() > 0
        if up2:
            condition &= df["ma2"].diff() > 0
        if up3:
            condition &= df["ma3"].diff() > 0

        signal = pd.Series(0, index=df.index, name="ema_signal")
        signal[condition] = 1

        # Décalage pour éviter le lookahead bias
        signal = signal.shift(1)

        return signal


    def generate_short(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)


        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        condition = (df["ma1"] < df["ma2"]) & (df["ma1"] < df["ma3"])

        if up1:
            condition &= df["ma1"].diff() < 0
        if up2:
            condition &= df["ma2"].diff() < 0
        if up3:
            condition &= df["ma3"].diff() < 0

       
        signal = pd.Series(0, index=df.index, name="zlema_signal_short")
        signal[condition] = 1

        signal = signal.shift(1)

        # to make dataframe
        result = pd.DataFrame(index=df.index)

        result["zma1"] = ma1
        result["zma2"] = ma2
        result["zma3"] = ma3

        result["zlema_signal"] = signal

        return signal, result
   
    def generate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)

        ma1_up = ma1.diff() > 0
        signal = (ma1 > ma2) & (ma1 > ma3) & ma1_up
        signal = signal.astype(int)

        result = pd.DataFrame(index=df.index)
        result["zma1"] = ma1
        result["zma2"] = ma2
        result["zma3"] = ma3
        result["zlema_signal"] = signal

        return result

   
def detect_successive_uptrend_from_close(close: pd.Series, min_successive_up: int = 3) -> pd.Series:
    s = close.copy()
    returns = s.diff()
    is_up = returns > 0

    # Détection avec rolling: 1 si les n dernières sont True
    successive = is_up.rolling(min_successive_up).apply(lambda x: all(x), raw=True)
    trend_active = successive.notna() & (successive == 1)

    result = pd.Series(index=close.index, dtype=object)
    result[trend_active] = 1

    # ⚠️ Décalage du signal pour prise de position à la bougie suivante
    result = result.shift(1)

    return result

def detect_successive_downtrend_from_close(close: pd.Series, min_successive_up: int = 3) -> pd.Series:
    s = close.copy()
    returns = s.diff()
    is_down = returns < 0

    # Détection avec rolling: 1 si les n dernières sont True
    successive = is_down.rolling(min_successive_up).apply(lambda x: all(x), raw=True)
    trend_active = successive.notna() & (successive == 1)

    result = pd.Series(index=close.index, dtype=object)
    result[trend_active] = 1
    # ⚠️ Décalage du signal pour prise de position à la bougie suivante
    result = result.shift(1)

    return result


def plot_signal(df, indicator_cols, signal_col, n_last=300, n_first=None):
    if n_first is not None:
        df_plot = df.head(n_first)
    else:
        df_plot = df.tail(n_last)

    plt.figure(figsize=(14, 6))

    for col in indicator_cols:
        plt.plot(df_plot.index, df_plot[col], label=col, marker='o', markersize=2, linewidth=1)

    signal_points = df_plot[df_plot[signal_col] == 1]
   
    if len(signal_points) > 0:
        first_indicator = indicator_cols[0]
        plt.scatter(signal_points.index, signal_points[first_indicator],
                    marker="^", color="red", label=signal_col, zorder=5)

    plt.title("indicateurs et signal")
    plt.xlabel("date")
    plt.ylabel("valeur")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_with_slider_window(df, indicator_cols, signal_col, n_total=500, window_size=10):
    df_total = df.head(n_total)

    max_pos = max(1, n_total - window_size)

    slider = widgets.IntSlider(min=0, max=max_pos, step=1, value=0, description='Position')

    def plot_window(pos):
        plt.figure(figsize=(14,6))
        df_window = df_total.iloc[pos : pos + window_size]

        for col in indicator_cols:
            plt.plot(df_window.index, df_window[col], label=col, marker='o', markersize=2, linewidth=1)

        signal_points = df_window[df_window[signal_col] == 1]
        if len(signal_points) > 0:
            first_indicator = indicator_cols[0]
            plt.scatter(signal_points.index, signal_points[first_indicator], marker='^', color='green', label=signal_col, zorder=5, s=100)

        plt.title(f"fenêtre de taille {window_size} à partir de la position {pos}")
        plt.xlabel("date")
        plt.ylabel("valeur")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import numpy as np




class Signal(ABC):
    @abstractmethod
    def generate(self, df) -> pd.Series:
        """
        Prend un dictionnaire de DataFrames (ex: {"5m": df_5min, "1h": df_1h})
        et retourne une Series booléenne de signaux alignée sur df["5m"]
        """
        pass

def compute_log_atr(data, lookback):
    # Log des prix
    log_high = np.log(data['high'])
    log_low = np.log(data['low'])
    log_close = np.log(data['close'])

    # Calcul du True Range en log
    prev_close = log_close.shift(1)
    tr1 = log_high - log_low
    tr2 = (log_high - prev_close).abs()
    tr3 = (log_low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Moyenne mobile simple du True Range = ATR
    atr = true_range.rolling(window=lookback).mean()

    return atr

class ZLEMATriple:
    def __init__(self, period_fast: int = 8, period_medium: int = 21, period_slow: int = 55, use_kalman: bool = True):
        self.period_fast = period_fast
        self.period_medium = period_medium
        self.period_slow = period_slow
        self.use_kalman = use_kalman

    def zlema(self, series: pd.Series, period: int) -> pd.Series:
        lag = int(round((period - 1) / 2))
        ema_input = series + (series - series.shift(lag))
        return ema_input.ewm(span=period, adjust=False).mean()

    def kalman_filter(self, series: pd.Series) -> pd.Series:
        value1 = pd.Series(index=series.index, dtype=float)
        value2 = pd.Series(index=series.index, dtype=float)
        value3 = pd.Series(index=series.index, dtype=float)
        tr = series.diff().abs()

        for i in range(1, len(series)):
            v1 = 0.2 * (series.iloc[i] - series.iloc[i-1]) + 0.8 * (value1.iloc[i-1] if pd.notna(value1.iloc[i-1]) else 0)
            v2 = 0.1 * tr.iloc[i] + 0.8 * (value2.iloc[i-1] if pd.notna(value2.iloc[i-1]) else tr.iloc[i])
            lambd = abs(v1 / v2) if v2 != 0 else 0.0001
            alpha = (-lambd**2 + np.sqrt(lambd**4 + 16 * lambd**2)) / 8
            v3 = alpha * series.iloc[i] + (1 - alpha) * (value3.iloc[i-1] if pd.notna(value3.iloc[i-1]) else series.iloc[i])

            value1.iloc[i] = v1
            value2.iloc[i] = v2
            value3.iloc[i] = v3

        return value3
   
   
    def generate(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)


        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        # condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])
        crossover = (df["ma1"] > df["ma2"]) & (df["ma1"].shift(1) <= df["ma2"].shift(1))
        # condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])
        condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])

        if up1:
            condition &= df["ma1"].diff() > 0
            # condition &=  np.gradient(ma1) > 0
        if up2:
            condition &= df["ma2"].diff() > 0
        if up3:
            condition &= df["ma3"].diff() > 0

        condition &= crossover

        signal = pd.Series(0, index=df.index, name="zlema_signal")
        signal[condition] = 1

        signal = signal.shift(1)

        return signal
   
    def generate_ema_signal(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
       
        # Source de prix : prix typique
        src = (df['high'] + df['low'] + df['close']) / 3

        # Filtrage optionnel par Kalman
        if self.use_kalman:
            src = self.kalman_filter(src)

        # Moyennes mobiles exponentielles
        ma1 = src.ewm(span=self.period_fast, adjust=False).mean()
        ma2 = src.ewm(span=self.period_medium, adjust=False).mean()
        ma3 = src.ewm(span=self.period_slow, adjust=False).mean()

        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        # Condition d'ordre croissant des EMAs
        condition = (df["ma1"] > df["ma2"]) & (df["ma1"] > df["ma3"])

        if up1:
            condition &= df["ma1"].diff() > 0
        if up2:
            condition &= df["ma2"].diff() > 0
        if up3:
            condition &= df["ma3"].diff() > 0

        signal = pd.Series(0, index=df.index, name="ema_signal")
        signal[condition] = 1

        # Décalage pour éviter le lookahead bias
        signal = signal.shift(1)

        return signal


    def generate_short(self, df, up1=True, up2=True, up3=True) -> pd.Series:
        df = df.copy()
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)


        df["ma1"] = ma1
        df["ma2"] = ma2
        df["ma3"] = ma3

        condition = (df["ma1"] < df["ma2"]) & (df["ma1"] < df["ma3"])

        if up1:
            condition &= df["ma1"].diff() < 0
        if up2:
            condition &= df["ma2"].diff() < 0
        if up3:
            condition &= df["ma3"].diff() < 0

       
        signal = pd.Series(0, index=df.index, name="zlema_signal_short")
        signal[condition] = 1

        signal = signal.shift(1)

        # to make dataframe
        result = pd.DataFrame(index=df.index)

        result["zma1"] = ma1
        result["zma2"] = ma2
        result["zma3"] = ma3

        result["zlema_signal"] = signal

        return signal, result
   
    def generate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        src = (df['high'] + df['low'] + df['close']) / 3

        if self.use_kalman:
            src = self.kalman_filter(src)

        ma1 = self.zlema(src, self.period_fast)
        ma2 = self.zlema(src, self.period_medium)
        ma3 = self.zlema(src, self.period_slow)

        ma1_up = ma1.diff() > 0
        signal = (ma1 > ma2) & (ma1 > ma3) & ma1_up
        signal = signal.astype(int)

        result = pd.DataFrame(index=df.index)
        result["zma1"] = ma1
        result["zma2"] = ma2
        result["zma3"] = ma3
        result["zlema_signal"] = signal

        return result

   
def detect_successive_uptrend_from_close(close: pd.Series, min_successive_up: int = 3) -> pd.Series:
    s = close.copy()
    returns = s.diff()
    is_up = returns > 0

    # Détection avec rolling: 1 si les n dernières sont True
    successive = is_up.rolling(min_successive_up).apply(lambda x: all(x), raw=True)
    trend_active = successive.notna() & (successive == 1)

    result = pd.Series(index=close.index, dtype=object)
    result[trend_active] = 1

    # ⚠️ Décalage du signal pour prise de position à la bougie suivante
    result = result.shift(1)

    return result

def detect_successive_downtrend_from_close(close: pd.Series, min_successive_up: int = 3) -> pd.Series:
    s = close.copy()
    returns = s.diff()
    is_down = returns < 0

    # Détection avec rolling: 1 si les n dernières sont True
    successive = is_down.rolling(min_successive_up).apply(lambda x: all(x), raw=True)
    trend_active = successive.notna() & (successive == 1)

    result = pd.Series(index=close.index, dtype=object)
    result[trend_active] = 1
    # ⚠️ Décalage du signal pour prise de position à la bougie suivante
    result = result.shift(1)

    return result


def plot_signal(df, indicator_cols, signal_col, n_last=300, n_first=None):
    if n_first is not None:
        df_plot = df.head(n_first)
    else:
        df_plot = df.tail(n_last)

    plt.figure(figsize=(14, 6))

    for col in indicator_cols:
        plt.plot(df_plot.index, df_plot[col], label=col, marker='o', markersize=2, linewidth=1)

    signal_points = df_plot[df_plot[signal_col] == 1]
   
    if len(signal_points) > 0:
        first_indicator = indicator_cols[0]
        plt.scatter(signal_points.index, signal_points[first_indicator],
                    marker="^", color="red", label=signal_col, zorder=5)

    plt.title("indicateurs et signal")
    plt.xlabel("date")
    plt.ylabel("valeur")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_with_slider_window(df, indicator_cols, signal_col, n_total=500, window_size=10):
    df_total = df.head(n_total)

    max_pos = max(1, n_total - window_size)

    slider = widgets.IntSlider(min=0, max=max_pos, step=1, value=0, description='Position')

    def plot_window(pos):
        plt.figure(figsize=(14,6))
        df_window = df_total.iloc[pos : pos + window_size]

        for col in indicator_cols:
            plt.plot(df_window.index, df_window[col], label=col, marker='o', markersize=2, linewidth=1)

        signal_points = df_window[df_window[signal_col] == 1]
        if len(signal_points) > 0:
            first_indicator = indicator_cols[0]
            plt.scatter(signal_points.index, signal_points[first_indicator], marker='^', color='green', label=signal_col, zorder=5, s=100)

        plt.title(f"fenêtre de taille {window_size} à partir de la position {pos}")
        plt.xlabel("date")
        plt.ylabel("valeur")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    widgets.interact(plot_window, pos=slider)
    widgets.interact(plot_window, pos=slider)