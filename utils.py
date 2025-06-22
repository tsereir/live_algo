import ccxt
import time
from my_config import API_KEY, API_SECRET, PASSWORD

def create_bitget_exchange(sandbox=True): 
    bitget = ccxt.bitget({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': PASSWORD,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  
        }
    })

    if sandbox : 
        bitget.set_sandbox_mode(True)
    return bitget

def is_in_position(exchange, symbol):
    try:
        positions = exchange.fetch_positions([symbol])
        if not positions:
            return False

        position_info = positions[0]["info"]

        total = float(position_info.get("total", "0"))
        hold_side = position_info.get("holdSide", "").lower()

        return total > 0 and hold_side == "long"

    except Exception as e:
        print(f"Erreur lors du check de position : {e}")
        return False
    
def wait_for_next_candle(timeframe: str):
    """
    attend jusqu'à ce que la bougie actuellle soit finalisée.
    """
    interval_sec = timeframe_to_seconds(timeframe)
    now = time.time()
    seconds_since_last = now % interval_sec
    seconds_to_next = interval_sec - seconds_since_last
    print(f"⏳ Attente {round(seconds_to_next, 2)} sec pour la prochaine bougie finalisée")
    time.sleep(seconds_to_next)

def timeframe_to_seconds(timeframe):
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == 's':
        return value
    elif unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")