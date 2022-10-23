
# random forest, linear regression, decisive tree, long short-term memory
from binance import Client
from Secret import api_key, secret_key
from Indicators import Indicators
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Binance
client = Client(api_key, secret_key)
symbol = "BTCUSDT"
interval = '5m'
klines = client.get_historical_klines(symbol, interval, "23 Oct,2022")
for line in klines:
    del line[5:]
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close']
data.open_time = [dt.datetime.fromtimestamp(x/1000.0) for x in data.open_time]
indicators = Indicators(data, [20, 50, 100])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.85, 0.15]
)

fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.rsi_result[100:],
    name='rsi',  row=2, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.bb_result[0][100:],
    name='bollinger_band up',  row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.bb_result[1][100:],
    name='bollinger_band down', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[20][100:],
    name='SMA 20',row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[50][100:],
    name='SMA 50',row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[100][100:],
    name='SMA 100',row=1, col=1)

fig.add_candlestick(x=data['open_time'][100:],
                                   open=data['open'][100:],
                                   high=data['high'][100:],
                                   low=data['low'][100:],
                                   close=data['close'][100:],
name='Market price',
    row=1, col=1
)
fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
fig.show()


def main():
    pass


if __name__ == "__main__":
    main()
