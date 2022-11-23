import pandas as pd
import numpy as np


class Indicators:
    def __init__(self, data, sma_periods):
        self.data = data
        self.sma_periods = sma_periods
        self.sma_result = self.simple_moving_average(sma_periods)
        self.bb_result = self.bollinger_bands()
        self.rsi_result = self.relative_strength_index()
        self.cci_result = self.commodity_channel_index()
        self.ichimoku_cloud_result = self.ichimoku_cloud()

    def simple_moving_average(self, sma_periods: list):
        a = list()
        for i in sma_periods:
            a.append(self.data['open'].rolling(window=i).mean())
        result = pd.DataFrame(a).transpose()
        result.columns = sma_periods
        return result

    def bollinger_bands(self):
        standard_dev = self.data['close'].rolling(20).std()
        bb_up = self.sma_result[20] + standard_dev * 2
        bb_down = self.sma_result[20] - standard_dev * 2
        return bb_up, bb_down

    def relative_strength_index(self, periods=14):
        to_float = self.data.apply(pd.to_numeric, errors='coerce')
        delta = to_float['close'].diff()
        up_results = delta.clip(lower=0)
        down_results = -1 * delta.clip(upper=0)
        ma_up = up_results.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down_results.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        rsi = 100 - (100 / (1 + (ma_up / ma_down)))
        return rsi

    def commodity_channel_index(self, periods=20):
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma_tp = tp.rolling(periods).mean()
        mad = tp.rolling(periods).apply(lambda x: pd.Series(x).mad())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    def ichimoku_cloud(self):
        elements = pd.DataFrame(columns=['Tenkan-sen', 'Kijun-sen', 'Senkou Span A', 'Senkou Span B', 'chikou_span'])
        period9_high = self.data['high'].rolling(window=9).max()
        period9_low = self.data['low'].rolling(window=9).min()
        elements['Tenkan-sen'] = (period9_high + period9_low) / 2

        period26_high = self.data['high'].rolling(window=26).max()
        period26_low = self.data['low'].rolling(window=26).min()
        elements['Kijun-sen'] = (period26_high + period26_low) / 2

        elements['Senkou Span A'] = ((elements['Tenkan-sen'] + elements['Kijun-sen']) / 2)

        period52_high = self.data['high'].rolling(window=52).max()
        period52_low = self.data['low'].rolling(window=52).min()
        elements['Senkou Span B'] = ((period52_high + period52_low) / 2)
        for i in range(0, 26):
            elements.loc[len(elements)] = ['nan', 'nan', 'nan', 'nan', 'nan']
        elements['Senkou Span B'] = elements['Senkou Span B'].shift(26)
        elements['Senkou Span A'] = elements['Senkou Span A'].shift(26)

        elements['chikou_span'] = self.data['close'].shift(-22)
        return elements


class IndicatorsAnalysis(Indicators):
    def __init__(self, data, sma_periods):
        super().__init__(data, sma_periods)
        self.sma_result['open_time'] = self.data['open_time']
        self.sma_result['sma_20_prev'] = self.sma_result[20].shift(1)
        self.sma_result['sma_50_prev'] = self.sma_result[50].shift(1)
        self.sma_result['sma_100_prev'] = self.sma_result[100].shift(1)
        # self.sma_analysis_results = self.sma_analysis()
        self.sma_result = self.sma_result.fillna(0)
        self.sma_result['crossovers'] = np.vectorize(self.sma_analysis)(self.sma_result[20],
                                                                        self.sma_result['sma_20_prev'],
                                                                        self.sma_result[50],
                                                                        self.sma_result['sma_50_prev'],
                                                                        self.sma_result[100],
                                                                        self.sma_result['sma_100_prev'])
        self.signal = self.sma_result[self.sma_result['crossovers'] == '+'].copy()

    def sma_analysis(self, sma_20, sma_20_prev, sma_50, sma_50_prev, sma_100, sma_100_prev, status="?"):
        if sma_20 > sma_50 > sma_20_prev and sma_20_prev < sma_50_prev \
                or sma_20 > sma_100 > sma_20_prev and sma_20_prev < sma_100_prev \
                or sma_50 > sma_100 > sma_50_prev and sma_50_prev < sma_100_prev:
            status = "+"
            return status
        if sma_20 < sma_50 < sma_20_prev and status != "-" \
                or sma_20 < sma_100 < sma_20_prev and status != "-" \
                or sma_50 < sma_100 < sma_50_prev and status != "-":
            status = "-"
            return status
        return None
