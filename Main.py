# random forest, linear regression, decisive tree, long short-term memory
from binance import Client
from Secret import _api_key, _secret_key
from Indicators import Indicators
import pandas as pd
import datetime as dt
from plotly.subplots import make_subplots
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output


# def update_plot():
#     klines = client.get_historical_klines(symbol, interval, "17 Nov,2022")
#     for line in klines:
#         del line[5:]
#     data = pd.DataFrame(klines)
#     data.columns = ['open_time', 'open', 'high', 'low', 'close']
#     data.open_time = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.open_time]
#     data = data.astype({"open": float, "high": float, "low": float, "close": float})
#     indicators = Indicators(data, [20, 50, 100])
#     fig.add_candlestick(x=data['open_time'][100:],
#                         open=data['open'][100:],
#                         high=data['high'][100:],
#                         low=data['low'][100:],
#                         close=data['close'][100:],
#                         name='Market price',
#                         row=1, col=1
#                         )


def draw_fig(ffig, fx, fy, fname, frow=1, fcol=1):
    ffig.add_scatter(
        x=fx,
        y=fy,
        name=fname, row=frow, col=fcol)


app = dash.Dash(__name__)
# Binance
client = Client(_api_key, _secret_key)
symbol = "BTCUSDT"
interval = '5m'
klines = client.get_historical_klines(symbol, interval, "17 Nov,2022")
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
    add_time.append(add_time[i] + dt.timedelta(minutes=5))
add_series = pd.Series(add_time)
b = add_series.values.astype('datetime64[m]')
a = data['open_time'][100:].values.astype('datetime64[m]')
c = a.tolist()
d = b.tolist()
e = c + d
f = pd.Series(e)


draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['Tenkan-sen'][100:], 'Tenkan-sen')
draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['Kijun-sen'][100:], 'Kijun-sen')
draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span A'][100:], 'Senkou Span A')
draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span B'][100:], 'Senkou Span B')
draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['chikou_span'][100:], 'chikou_span')
draw_fig(fig, data['open_time'][100:], indicators.cci_result[100:], 'CCI', frow=3)
draw_fig(fig, data['open_time'][100:], indicators.rsi_result[100:], 'RSI', frow=2)
draw_fig(fig, data['open_time'][100:], indicators.bb_result[0][100:], 'bollinger_band up')
draw_fig(fig, data['open_time'][100:], indicators.bb_result[1][100:], 'bollinger_band down')
draw_fig(fig, data['open_time'][100:], indicators.sma_result[20][100:], 'SMA 20')
draw_fig(fig, data['open_time'][100:], indicators.sma_result[50][100:], 'SMA 50')
draw_fig(fig, data['open_time'][100:], indicators.sma_result[100][100:], 'SMA 100')
fig.add_candlestick(x=data['open_time'][100:],
                    open=data['open'][100:],
                    high=data['high'][100:],
                    low=data['low'][100:],
                    close=data['close'][100:],
                    name='Market price',
                    row=1, col=1
                    )
fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark")
# fig.show()

app.layout = html.Div([
    html.H1("Market Plot", style={'text-align': 'center'}),

    dcc.Graph(
        id='live-graph',
        figure=fig,
        style={'height': '90vh'},
        animate=True
    ),

    dcc.Interval(id='graph-update', interval=10_000)
])

# @@app.callback(Output('live-graph', 'figure'),
#               [Input('graph-update', 'n_intervals')])
# def update_graph_scatter(input_data):
#
#     data = fig
#     return {'data': [data],'layout' : fig}


if __name__ == '__main__':
    app.run_server(debug=True)
