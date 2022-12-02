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
        result = pd.DataFrame(a).transpose().shift(-1)
        result.columns = sma_periods
        return result

    def bollinger_bands(self):
        standard_dev = self.data['close'].rolling(20).std()
        bb_up = pd.Series(self.sma_result[20] + standard_dev * 2)
        bb_down = pd.Series(self.sma_result[20] - standard_dev * 2)
        return pd.concat([bb_up, bb_down], axis=1)

    def relative_strength_index(self, periods=14):
        to_float = self.data.apply(pd.to_numeric, errors='coerce')
        delta = to_float['close'].diff()
        up_results = delta.clip(lower=0)
        down_results = -1 * delta.clip(upper=0)
        ma_up = up_results.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down_results.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        rsi = 100 - (100 / (1 + (ma_up / ma_down)))
        return pd.Series(rsi)

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
        self.data = data
        self.sma_result['open_time'] = self.data['open_time']
        self.sma_result['close'] = self.data['close']
        self.sma_result['open'] = self.data['open']
        self.sma_result['sma_20_prev'] = self.sma_result[20].shift(1)
        self.sma_result['sma_50_prev'] = self.sma_result[50].shift(1)
        self.sma_result['sma_100_prev'] = self.sma_result[100].shift(1)
        self.sma_result['close_prev'] = self.data['close'].shift(1)
        self.sma_result['bb_up_prev'] = self.bb_result[0].shift(1)
        self.sma_result['bb_down_prev'] = self.bb_result[1].shift(1)
        self.sma_result['rsi_prev'] = self.rsi_result.shift(1)
        self.sma_result = self.sma_result.fillna(0)
        self.sma_result['crossovers'] = np.vectorize(self.sma_cross_analysis)(self.sma_result[20],
                                                                              self.sma_result['sma_20_prev'],
                                                                              self.sma_result[50],
                                                                              self.sma_result['sma_50_prev'],
                                                                              self.sma_result[100],
                                                                              self.sma_result['sma_100_prev'])

        self.sma_result['50_to_100'] = np.vectorize(self.sma_50_to_100)(self.sma_result[100],
                                                                        self.sma_result[50],
                                                                        self.data['close'],
                                                                        self.sma_result['sma_50_prev'],
                                                                        self.sma_result['close_prev'],
                                                                        self.sma_result['sma_100_prev'])

        self.sma_result['sma_cross_rsi'] = np.vectorize(self.sma_cross_rsi)(self.sma_result[20],
                                                                            self.sma_result['sma_20_prev'],
                                                                            self.sma_result[50],
                                                                            self.sma_result['sma_50_prev'],
                                                                            self.sma_result[100],
                                                                            self.sma_result['sma_100_prev'],
                                                                            self.rsi_result,
                                                                            self.sma_result['rsi_prev'])

        self.sma_result['bb_squezze'] = np.vectorize(self.bb_squeeze)(self.bb_result[0],
                                                                      self.bb_result[0].shift(1),
                                                                      self.bb_result[1],
                                                                      self.bb_result[1].shift(1),
                                                                      self.sma_result[20],
                                                                      self.data['close'])

        self.sma_result['cci_100_bb_cross'] = np.vectorize(self.cci_100_bb_cross)(self.data['open'],
                                                                                  self.data['close'],
                                                                                  self.sma_result[20],
                                                                                  self.sma_result['close_prev'],
                                                                                  self.cci_result)

        self.sma_result['rsi_chart'] = np.vectorize(self.rsi_chart)(self.rsi_result,
                                                                    self.rsi_result.shift(1))

        self.sma_result['rsi_bb_boundary'] = np.vectorize(self.rsi_bb_boundary)(self.rsi_result,
                                                                                self.rsi_result.shift(1),
                                                                                self.bb_result[0],
                                                                                self.bb_result[1],
                                                                                self.data['open'],
                                                                                self.data['close']
                                                                                )

        self.signal_up = self.sma_result[self.sma_result['crossovers'] == "Buy"].copy()
        self.signal_down = self.sma_result[self.sma_result['crossovers'] == "Sell"].copy()

        self.signal_50_to_100_up = self.sma_result[self.sma_result['50_to_100'] == "Buy"].copy()
        self.signal_50_to_100_down = self.sma_result[self.sma_result['50_to_100'] == "Sell"].copy()

        self.sma_cross_rsi_up = self.sma_result[self.sma_result['sma_cross_rsi'] == "Buy"].copy()
        self.sma_cross_rsi_down = self.sma_result[self.sma_result['sma_cross_rsi'] == "Sell"].copy()

        self.signal_bb_squeeze_up = self.sma_result[self.sma_result['bb_squezze'] == "Buy"].copy()
        self.signal_bb_squeeze_down = self.sma_result[self.sma_result['bb_squezze'] == "Sell"].copy()

        self.cci_100_bb_cross_up = self.sma_result[self.sma_result['cci_100_bb_cross'] == "Buy"].copy()
        self.cci_100_bb_cross_down = self.sma_result[self.sma_result['cci_100_bb_cross'] == "Sell"].copy()

        self.rsi_chart_up = self.sma_result[self.sma_result['rsi_chart'] == "Buy"].copy()
        self.rsi_chart_down = self.sma_result[self.sma_result['rsi_chart'] == "Sell"].copy()

        self.rsi_bb_boundary_up = self.sma_result[self.sma_result['rsi_bb_boundary'] == "Buy"].copy()
        self.rsi_bb_boundary_down = self.sma_result[self.sma_result['rsi_bb_boundary'] == "Sell"].copy()
        ...

    def sma_cross_analysis(self, sma_20, sma_20_prev, sma_50, sma_50_prev, sma_100, sma_100_prev):
        if sma_20 > sma_50 > sma_20_prev and sma_20_prev < sma_50_prev \
                or sma_20 > sma_100 > sma_20_prev and sma_20_prev < sma_100_prev \
                or sma_50 > sma_100 > sma_50_prev and sma_50_prev < sma_100_prev:
            status = "Buy"
            return status
        if sma_20 < sma_50 < sma_20_prev and sma_20_prev > sma_50_prev \
                or sma_20 < sma_100 < sma_20_prev and sma_20_prev > sma_100_prev \
                or sma_50 < sma_100 < sma_50_prev and sma_50_prev > sma_100_prev:
            status = "Sell"
            return status
        return None

    def sma_50_to_100(self, sma_100, sma_50, close, sma_50_prev, close_prev, sma_100_prev):
        if sma_100 > sma_50 and abs(sma_100 - sma_50) > (0.015 * close) and sma_50 < close and sma_50_prev > close_prev:
            return "Buy"
        elif sma_50 < sma_100 <= close and abs(sma_100 - sma_50) > (0.008 * close) and sma_100_prev > close_prev:
            return "Sell"
        return None

    def sma_cross_rsi(self, sma_20, sma_20_prev, sma_50, sma_50_prev, sma_100, sma_100_prev, rsi, rsi_prev):
        if sma_20 > sma_50 > sma_20_prev and sma_20_prev < sma_50_prev and rsi > 60 \
                or sma_20 > sma_100 > sma_20_prev and sma_20_prev < sma_100_prev and rsi > 60 \
                or sma_50 > sma_100 > sma_50_prev and sma_50_prev < sma_100_prev and rsi > 60:
            status = "Buy"
            return status
        if rsi < 50 < rsi_prev:
            status = "Sell"
            return status
        return None

    def bb_squeeze(self, bb_result_up, bb_result_up_prev, bb_result_down, bb_result_down_prev, sma_20, close):
        if abs(bb_result_up_prev - bb_result_down_prev) < (0.020 * sma_20) \
                and (0.022 * sma_20) < abs(bb_result_up - bb_result_down) \
                and close > bb_result_up:
            return "Buy"
        elif abs(bb_result_up_prev - bb_result_down_prev) < (0.020 * sma_20) \
                and (0.022 * sma_20) < abs(bb_result_up - bb_result_down) \
                and close < bb_result_up:
            return "Sell"
        return None

    def cci_100_bb_cross(self, open_price, close, sma_20, close_prev, cci):
        if abs(open_price - close) > 125 and close > open_price and abs(
                sma_20 - close) > 125 / 3 and close_prev < sma_20 \
                and cci > 100:
            return "Buy"
        if abs(open_price - close) > 125 and close < open_price and abs(
                sma_20 - close) > 125 / 3 and close_prev > sma_20 \
                and cci < -40:
            return "Sell"
        return None

    def rsi_chart(self, rsi, rsi_prev):
        if 70 < rsi < rsi_prev and abs(rsi - rsi_prev) > 4:
            return "Sell"
        if 30 > rsi > rsi_prev and abs(rsi - rsi_prev) > 4:
            return "Buy"
        return None

    def rsi_bb_boundary(self, rsi, rsi_prev, bb_top, bb_bottom, open_price, close):
        if 30 > rsi and abs(rsi - rsi_prev) > 2 and open_price > bb_bottom > close:
            return "Buy"
        if 70 < rsi and abs(rsi - rsi_prev) > 2 and open_price < bb_top < close:
            return "Sell"
        return None
