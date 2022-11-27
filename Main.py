# random forest, linear regression, decisive tree, long short-term memory
from binance import Client
from Secret import _api_key, _secret_key
from Indicators import Indicators
from Indicators import IndicatorsAnalysis
from Learning_model import LearningModel
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
interval = '1h'
klines = client.get_historical_klines(symbol, interval, "1 Sep,2022")
for line in klines:
    del line[5:]
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close']
data.open_time = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.open_time]
data = data.astype({"open": float, "high": float, "low": float, "close": float})
indicators = Indicators(data, [20, 50, 100])
indicators_analysis = IndicatorsAnalysis(data, [20, 50, 100])
learning_model = LearningModel(indicators_analysis)

strategies = ["crossovers", "50_to_100", "sma_cross_rsi", "bb_squezze", "cci_100_bb_cross", "rsi_chart", "rsi_bb_boundary"]
for strategy in strategies:
    learning_model.testing(strategy)
...

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.80, 0.10, 0.10]
)
add_time = list()
add_time.append(data.tail(1)['open_time'].item())
for i in range(0, 26):
    add_time.append(add_time[i] + dt.timedelta(hours=1))
add_series = pd.Series(add_time)
b = add_series.values.astype('datetime64[m]')
a = data['open_time'][100:].values.astype('datetime64[m]')
c = a.tolist()
d = b.tolist()
e = c + d
f = pd.Series(e)
#
# for i, row in indicators_analysis.signal.iterrows():
#     fig.add_scatter(x=data['open_time'][i], y = indicators.sma_result[20][i], name= 'dupa', marker=dict(color="gold", size=12), row=1, col=1)
# fig.add_scatter( x=indicators_analysis.signal_up['open_time'], y = indicators_analysis.signal_up[20], mode ='markers',
#                  name= 'dupa', marker=dict(color="gold", size = 12, ), row=1, col=1)
# fig.add_scatter( x=indicators_analysis.signal_down['open_time'], y = indicators_analysis.signal_down[50], mode ='markers',
#                  name= 'dupa', marker=dict(color="red", size = 12, ), row=1, col=1)
#
# draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['Tenkan-sen'][100:], 'Tenkan-sen')
# draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['Kijun-sen'][100:], 'Kijun-sen')
# draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span A'][100:], 'Senkou Span A')
# draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span B'][100:], 'Senkou Span B')
# draw_fig(fig, data['open_time'][100:], indicators.ichimoku_cloud_result['chikou_span'][100:], 'chikou_span')
# draw_fig(fig, data['open_time'][100:], indicators.cci_result[100:], 'CCI', frow=3)
# draw_fig(fig, data['open_time'][100:], indicators.rsi_result[100:], 'RSI', frow=2)
# draw_fig(fig, data['open_time'][100:], indicators.bb_result[0][100:], 'bollinger_band up')
# draw_fig(fig, data['open_time'][100:], indicators.bb_result[1][100:], 'bollinger_band down')
# draw_fig(fig, data['open_time'][100:], indicators.sma_result[20][100:], 'SMA 20')
# draw_fig(fig, data['open_time'][100:], indicators.sma_result[50][100:], 'SMA 50')
# draw_fig(fig, data['open_time'][100:], indicators.sma_result[100][100:], 'SMA 100')
# fig.add_candlestick(x=data['open_time'][100:],
#                     open=data['open'][100:],
#                     high=data['high'][100:],
#                     low=data['low'][100:],
#                     close=data['close'][100:],
#                     name='Market price',
#                     row=1, col=1
#                     )


# SMA CROSS
fig.add_scatter( x=indicators_analysis.signal_up['open_time'], y = indicators_analysis.signal_up[20], mode ='markers',
                 name= 'signal_up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_down['open_time'], y = indicators_analysis.signal_down[50], mode ='markers',
                 name= 'signal_down', marker=dict(color="red", size = 12, ), row=1, col=1)
# SMA 50 TO 100
fig.add_scatter( x=indicators_analysis.signal_50_to_100_down['open_time'], y = indicators_analysis.signal_50_to_100_down[100], mode ='markers',
                 name= 'signal_50_to_100_down', marker=dict(color="red", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_50_to_100_up['open_time'], y = indicators_analysis.signal_50_to_100_up[100], mode ='markers',
                 name= 'signal_50_to_100_up', marker=dict(color="green", size = 12, ), row=1, col=1)
# # RSI AND SMA
# fig.add_scatter( x=indicators_analysis.sma_cross_rsi_up['open_time'], y = indicators_analysis.sma_cross_rsi_up[20], mode ='markers',
#                  name= 'sma_cross_rsi', marker=dict(color="green", size = 12, ), row=1, col=1)
# fig.add_scatter( x=indicators_analysis.sma_cross_rsi_down['open_time'], y = indicators_analysis.sma_cross_rsi_down[20], mode ='markers',
#                  name= 'sma_cross_rsi', marker=dict(color="red", size = 12, ), row=1, col=1)
# BB SQUEEZE
fig.add_scatter( x=indicators_analysis.signal_bb_squeeze_up['open_time'], y = indicators_analysis.signal_bb_squeeze_up[20], mode ='markers',
                 name= 'signal_bb_squeeze_up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_bb_squeeze_down['open_time'], y = indicators_analysis.signal_bb_squeeze_down[20], mode ='markers',
                 name= 'signal_bb_squeeze_down', marker=dict(color="red", size = 12, ), row=1, col=1)
# CCI 100 & BB CROSS
fig.add_scatter( x=indicators_analysis.cci_100_bb_cross_up['open_time'], y = indicators_analysis.cci_100_bb_cross_up[20], mode ='markers',
                 name= 'cci_100_bb_cross', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.cci_100_bb_cross_down['open_time'], y = indicators_analysis.cci_100_bb_cross_down[20], mode ='markers',
                 name= 'cci_100_bb_cross', marker=dict(color="red", size = 12, ), row=1, col=1)
# RSI
fig.add_scatter( x=indicators_analysis.rsi_chart_up['open_time'], y = indicators_analysis.rsi_chart_up[20], mode ='markers',
                 name= 'rsi_chart', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.rsi_chart_down['open_time'], y = indicators_analysis.rsi_chart_down[20], mode ='markers',
                 name= 'rsi_chart', marker=dict(color="red", size = 12, ), row=1, col=1)
# RSI AND BB
fig.add_scatter( x=indicators_analysis.rsi_bb_boundary_up['open_time'], y = indicators_analysis.rsi_bb_boundary_up[20], mode ='markers',
                 name= 'rsi_bb_boundary', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.rsi_bb_boundary_down['open_time'], y = indicators_analysis.rsi_bb_boundary_down[20], mode ='markers',
                 name= 'rsi_bb_boundary', marker=dict(color="red", size = 12, ), row=1, col=1)






draw_fig(fig, data['open_time'], indicators.ichimoku_cloud_result['Tenkan-sen'], 'Tenkan-sen')
draw_fig(fig, data['open_time'], indicators.ichimoku_cloud_result['Kijun-sen'], 'Kijun-sen')
draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span A'], 'Senkou Span A')
draw_fig(fig, f, indicators.ichimoku_cloud_result['Senkou Span B'], 'Senkou Span B')
draw_fig(fig, data['open_time'], indicators.ichimoku_cloud_result['chikou_span'], 'chikou_span')
draw_fig(fig, data['open_time'], indicators.cci_result, 'CCI', frow=3)
draw_fig(fig, data['open_time'], indicators.rsi_result, 'RSI', frow=2)
draw_fig(fig, data['open_time'], indicators.bb_result[0], 'bollinger_band up')
draw_fig(fig, data['open_time'], indicators.bb_result[1], 'bollinger_band down')
draw_fig(fig, data['open_time'], indicators.sma_result[20], 'SMA 20')
draw_fig(fig, data['open_time'], indicators.sma_result[50], 'SMA 50')
draw_fig(fig, data['open_time'], indicators.sma_result[100], 'SMA 100')
fig.add_candlestick(x=data['open_time'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Market price',
                    row=1, col=1
                    )
fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", )
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
