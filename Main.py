
# random forest, linear regression, decisive tree, long short-term memory


from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from Secret import api_key, secret_key
from Indicators import Indicators
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import mplfinance as mpf
from tradingview_ta import TA_Handler, Interval, Exchange
from tvDatafeed import TvDatafeed, Interval
import numpy as np
# TradingView
# tv = TvDatafeed()
# nifty_index_data = tv.get_hist(symbol='BTCUSDT', exchange='Binance', interval=Interval.in_5_minute, n_bars=1000)
# handler = TA_Handler(
#     symbol="BTCUSDT",
#     exchange="Binance",
#     screener="Crypto",
#     interval="5m",
#     timeout=None
# )
# a = handler.get_analysis().indicators

# Binance
client = Client(api_key, secret_key)
symbol = "BTCUSDT"
interval = '5m'
klines = client.get_historical_klines(symbol, interval, "20 Oct,2022")
for line in klines:
    del line[5:]
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close']
data.open_time = [dt.datetime.fromtimestamp(x/1000.0) for x in data.open_time]
resolution = int(float(data['high'].min())), int(float(data['high'].max()))
difference = abs(resolution[0]-resolution[1])
indicators = Indicators(data, [20, 50, 100])
fig, ax = plt.subplots()
data.index = pd.DatetimeIndex(data['open_time'])
data = data.drop('open_time', axis = 1)
#ax1.plot(data.open_time, indicators.sma_result)
data = data.apply(pd.to_numeric, errors='coerce')
up = data[data['close'] >= data['open']]

down = data[data['close'] < data['open']]
col1 = 'green'

col2 = 'red'

width = .0027
width2 = .00019
ax.set_facecolor('black')
ax.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
ax.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
ax.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)

ax.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
ax.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
ax.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)
plt.plot(data.index, indicators.sma_result)
ax.grid(color='gray')
ax.set_xlabel('time')
ax.set_ylabel('price')
ax.set_title("Bitcoin chart")
plt.show()
...

def main():
    pass


if __name__ == "__main__":
    main()

...