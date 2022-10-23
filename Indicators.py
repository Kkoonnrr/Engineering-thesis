import pandas as pd


class Indicators:
    def __init__(self, data, sma_periods):
        self.data = data
        self.sma_periods = sma_periods
        self.sma_result = self.simple_moving_average(sma_periods)
        self.bb_result = self.bollinger_bands()
        self.rsi_result = self.relative_strength_index()

    def simple_moving_average(self, sma_periods: list):
        a = list()
        for i in sma_periods:
            a.append(self.data['open'].rolling(window=i).mean())
        result = pd.DataFrame(a).transpose()
        result.columns = sma_periods
        return result

    def bollinger_bands(self):
        standard_dev = self.data['close'].rolling(20).std()
        bb_up = self.sma_result[20] + standard_dev * 2  # Calculate top band
        bb_down = self.sma_result[20] - standard_dev * 2  # Calculate bottom band
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

    def commodity_channel_index(self):
        pass

    def ichimoku_cloud(self):
        pass

