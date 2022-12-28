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
from itertools import combinations
from Learning_model import RegresjaLogistyczna


# Drawing plots
def draw_fig(ffig, fx, fy, fname, frow=1, fcol=1):
    ffig.add_scatter(
        x=fx,
        y=fy,
        name=fname, row=frow, col=fcol)


app = dash.Dash(__name__)
# Downloading Binance historic data
client = Client(_api_key, _secret_key)
symbol = "BTCUSDT"
interval = '1h'
klines = client.get_historical_klines(symbol, interval, "1 Jun, 2022")
# Creating DataFrame
for line in klines:
    del line[5:]
data = pd.DataFrame(klines)
data.columns = ['open_time', 'open', 'high', 'low', 'close']
data.open_time = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.open_time]
data = data.astype({"open": float, "high": float, "low": float, "close": float})
# Creating Indicator instance
indicators = Indicators(data, [20, 50, 100])
# Creating IndicatorAnalysis instance
indicators_analysis = IndicatorsAnalysis(data, [20, 50, 100])
# Creating LearningModel instance
learning_model = LearningModel(indicators_analysis)
# Creating combination of the strategies
strategies_comb = list()
strategies = ["crossovers", "50_to_100", "sma_cross_rsi", "bb_squezze", "cci_100_bb_cross", "rsi_chart", "rsi_bb_boundary"]
for i in range (1, 8):
    strategies_comb.append(combinations(strategies, i))
# Testing combination in simulation
for i in strategies_comb:
    for j in list(i):
        df = pd.DataFrame(learning_model.testing(j))
# Putting results of simulation into .csv file
df.to_csv("results.csv")
# Making plots
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
regresja = RegresjaLogistyczna(indicators_analysis.sma_result.drop([20,50,100], axis=1))
# SMA CROSS
fig.add_scatter( x=indicators_analysis.signal_up['open_time'], y = indicators_analysis.signal_up[20], mode ='markers',
                 name= 'SMA przecięcia up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_down['open_time'], y = indicators_analysis.signal_down[50], mode ='markers',
                 name= 'SMA przecięcia down', marker=dict(color="red", size = 12, ), row=1, col=1)
# SMA 50 TO 100
fig.add_scatter( x=indicators_analysis.signal_50_to_100_down['open_time'], y = indicators_analysis.signal_50_to_100_down[100], mode ='markers',
                 name= 'SMA 50 do SMA 100 down', marker=dict(color="red", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_50_to_100_up['open_time'], y = indicators_analysis.signal_50_to_100_up[100], mode ='markers',
                 name= 'SMA 50 do SMA 100 up', marker=dict(color="green", size = 12, ), row=1, col=1)
# RSI AND SMA
fig.add_scatter( x=indicators_analysis.sma_cross_rsi_up['open_time'], y = indicators_analysis.sma_cross_rsi_up[20], mode ='markers',
                 name= 'SMA przecięcia + RSI up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.sma_cross_rsi_down['open_time'], y = indicators_analysis.sma_cross_rsi_down[20], mode ='markers',
                 name= 'SMA przecięcia + RSI down', marker=dict(color="red", size = 12, ), row=1, col=1)
# BB SQUEEZE
fig.add_scatter( x=indicators_analysis.signal_bb_squeeze_up['open_time'], y = indicators_analysis.signal_bb_squeeze_up[20], mode ='markers',
                 name= 'BB ściśnięcia up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.signal_bb_squeeze_down['open_time'], y = indicators_analysis.signal_bb_squeeze_down[20], mode ='markers',
                 name= 'BB ściśnięcia down', marker=dict(color="red", size = 12, ), row=1, col=1)
# CCI 100 & BB CROSS
fig.add_scatter( x=indicators_analysis.cci_100_bb_cross_up['open_time'], y = indicators_analysis.cci_100_bb_cross_up[20], mode ='markers',
                 name= 'SMA 20 przecięcie + CCI up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.cci_100_bb_cross_down['open_time'], y = indicators_analysis.cci_100_bb_cross_down[20], mode ='markers',
                 name= 'SMA 20 przecięcie + CCI down', marker=dict(color="red", size = 12, ), row=1, col=1)
# RSI
fig.add_scatter( x=indicators_analysis.rsi_chart_up['open_time'], y = indicators_analysis.rsi_chart_up[20], mode ='markers',
                 name= 'RSI up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.rsi_chart_down['open_time'], y = indicators_analysis.rsi_chart_down[20], mode ='markers',
                 name= 'RSI down', marker=dict(color="red", size = 12, ), row=1, col=1)
# RSI AND BB
fig.add_scatter( x=indicators_analysis.rsi_bb_boundary_up['open_time'], y = indicators_analysis.rsi_bb_boundary_up[20], mode ='markers',
                 name= 'BB granice + RSI up', marker=dict(color="green", size = 12, ), row=1, col=1)
fig.add_scatter( x=indicators_analysis.rsi_bb_boundary_down['open_time'], y = indicators_analysis.rsi_bb_boundary_down[20], mode ='markers',
                 name= 'BB granice + RSI down', marker=dict(color="red", size = 12, ), row=1, col=1)

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
fig.update_xaxes(
        title_text = "CZAS")
fig.update_yaxes(
        title_text = "KWOTA")
# Displaying plots
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
# Start project
if __name__ == '__main__':
    app.run_server(debug=True)
