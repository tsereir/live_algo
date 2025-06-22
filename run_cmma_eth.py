import time
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from pprint import pprint

from utils import *
from market_data import *

from indicators import *

import my_config


def process_strategie_single(pair, 
                      exchange, 
                      timeframe="1m", 
                      maxlen=1000, 
                      pair2=None, 
                      leverage=3):
    
    print("🕒 Début exécution à :", datetime.now())

    exchange.set_leverage(leverage, pair2, params={"marginMode": "isolated"})

    price_cache_btc = fetch_latest_ohlcv_to_deque(pair, exchange, timeframe, maxlen=maxlen, only_close=False)
    price_cache_eth = fetch_latest_ohlcv_to_deque(pair2, exchange, timeframe, maxlen=maxlen, only_close=False)

    data_btc = pd.DataFrame(price_cache_btc, columns=["date", "open", "high", "low", "close", "volume"]).set_index("date")
    data_eth = pd.DataFrame(price_cache_eth, columns=["date", "open", "high", "low", "close", "volume"]).set_index("date")
    

    diff, sig = get_cmma_signal(data_btc,
                                data_eth,
                                lookback=my_config.SMA_LOOKBACK,
                                atr_lookback=my_config.ATR_LOOKBACK,
                                entry_threshold=my_config.ENTRY_THRESHOLD,
                                exit_threshold=my_config.EXIT_THRESHOLD)

    print("📈 Intermarket diff :", diff)
    print("📊 Signal :", sig)
    in_position = is_in_position(exchange=exchange, symbol=pair2)

    try:
        if not in_position and sig == 1:
            print("🟢 Entrée LONG")
            order = exchange.create_market_buy_order(pair2, 0.01, {"marginMode": "isolated"})
            print("✅ Ordre LONG :", order)

        elif in_position and sig == -1:
            print("🔴 Sortie LONG")
            order = exchange.create_market_sell_order(pair2, 0.01, {"reduceOnly": True, "marginMode": "isolated"})
            print("✅ Ordre de sortie :", order)

        else:
            print("⏸️ Aucune action nécessaire.")

    except Exception as e:
        print("❌ Erreur lors du passage d'ordre :", e)

if __name__ == "__main__":
    bitget = ccxt.bitget({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': PASSWORD,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  
        }
    })
    bitget.set_sandbox_mode(True)

    process_strategie_single(
        pair="BTCUSDT",
        exchange=bitget,
        timeframe="1m",
        maxlen=800,
        pair2="ETHUSDT",
        leverage=3
    )