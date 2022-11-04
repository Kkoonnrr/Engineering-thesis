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
klines = client.get_historical_klines(symbol, interval, "03 Nov,2022")
for line in klines:
    del line[5:]
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close']
data.open_time = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.open_time]
data = data.astype({"open": float, "high": float, "low": float, "close": float})
indicators = Indicators(data, [20, 50, 100])

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.80, 0.10, 0.10]
)
add_time = list()
add_time.append(data.tail(1)['open_time'].item())
for i in range(0, 26):
    add_time.append(add_time[i]+dt.timedelta(minutes = 5))
add_series = pd.Series(add_time)
b=add_series.values.astype('datetime64[m]')
a=data['open_time'][100:].values.astype('datetime64[m]')
c = a.tolist()
d = b.tolist()
e = c+d
f = pd.Series(e)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.ichimoku_cloud_result['Tenkan-sen'][100:],
    name='Tenkan-sen', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.ichimoku_cloud_result['Kijun-sen'][100:],
    name='Kijun-sen', row=1, col=1)
fig.add_scatter(
    x=f,
    y=indicators.ichimoku_cloud_result['Senkou Span A'][100:],
    name='Senkou Span A', row=1, col=1)
fig.add_scatter(
    x=f,
    y=indicators.ichimoku_cloud_result['Senkou Span B'][100:],
    name='Senkou Span B', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.ichimoku_cloud_result['chikou_span'][100:],
    name='chikou_span', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.cci_result[100:],
    name='CCI', row=3, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.rsi_result[100:],
    name='RSI', row=2, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.bb_result[0][100:],
    name='bollinger_band up', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.bb_result[1][100:],
    name='bollinger_band down', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[20][100:],
    name='SMA 20', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[50][100:],
    name='SMA 50', row=1, col=1)
fig.add_scatter(
    x=data['open_time'][100:],
    y=indicators.sma_result[100][100:],
    name='SMA 100', row=1, col=1)

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
