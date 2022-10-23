import pandas as pd
class Indicators:
    def __init__(self, data, sma_periods):
        self.data = data
        self.sma_periods = sma_periods
        self.sma_result = self.sma(sma_periods)

    def bollinger_bands(self):
        pass

    def sma(self, sma_periods: list):
        a = list()
        for i in sma_periods:
            a.append(self.data['open'].rolling(window=i).mean())
        result = pd.DataFrame(a).transpose()
        result.columns = sma_periods
        return result
