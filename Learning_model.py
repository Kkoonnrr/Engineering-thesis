from Indicators import Indicators
from Indicators import IndicatorsAnalysis
import pandas as pd


class LearningModel:
    def __init__(self, indicator_analysis: IndicatorsAnalysis):
        self.indicator_analysis = indicator_analysis
        self.results = pd.DataFrame(columns=["Strategy type", "% successful trades", "average income per trade",
                                             "maximum money", "minimum money", "money at the end"], ignore)

    def testing(self, given_strategy):
        start = 1000.0
        accumulated_crypto = 0.0
        average = 0.0
        maximum = start
        minimum = start
        self.results = self.results.append({"Strategy type": given_strategy})
        for index, row in self.indicator_analysis.sma_result.iterrows():
            if row[given_strategy] == "Buy":
                buy_amount = start * 0.1
                accumulated_crypto = accumulated_crypto + buy_amount / row['close']
                start = start - buy_amount
            elif row[given_strategy] == "Sell":
                sell_amount = accumulated_crypto * 0.5
                start = start + sell_amount * row['close']
                accumulated_crypto = accumulated_crypto - sell_amount
                if start > maximum:
                    maximum = start
                elif start < minimum:
                    minimum = start
            if index == len(self.indicator_analysis.sma_result) - 1:
                start = start + accumulated_crypto * row['close']
        self.results["maximum money"] = maximum
        self.results["minimum money"] = minimum
        self.results["money at the end"] = start
        print(f"Strategy: {given_strategy}")
        print(f"Money at the end: {start}")
        print("---------------------------------")
        # print(f"{accumulated_crypto}: ")
