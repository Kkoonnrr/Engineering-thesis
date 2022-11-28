from Indicators import Indicators
from Indicators import IndicatorsAnalysis
import pandas as pd


class LearningModel:
    def __init__(self, indicator_analysis: IndicatorsAnalysis):
        self.indicator_analysis = indicator_analysis
        self.results = {"Strategy type": [], "% successful trades": [], "average income per trade": [],
                                             "maximum money": [], "minimum money": [], "money at the end": [],
                        "buy operations": [], "sell operations": []}

    def testing(self, given_strategy):
        start = 1000.0
        accumulated_crypto = 0.0
        average = 0.0
        maximum = start
        minimum = start
        buy = 0
        sell = 0
        fee = 0.001
        actual = list()
        for x in given_strategy:
            actual.append(x)
        self.results['Strategy type'].append(actual)
        for index, row in self.indicator_analysis.sma_result.iterrows():
            for z in actual:
                if row[str(z)] == "Buy":
                    buy = buy + 1
                    buy_amount = start * 0.5
                    accumulated_crypto = accumulated_crypto + (buy_amount - buy_amount * fee) / row['close']
                    start = start - buy_amount
                elif row[str(z)] == "Sell":
                    sell = sell + 1
                    sell_amount = accumulated_crypto * 0.7
                    start = start + (sell_amount - sell_amount * fee) * row['close']
                    accumulated_crypto = accumulated_crypto - sell_amount
                    if start > maximum:
                        maximum = start
                    elif start < minimum:
                        minimum = start
                if index == len(self.indicator_analysis.sma_result) - 1:
                    start = start + accumulated_crypto * row['close']
                    if start > maximum:
                        maximum = start
                    elif start < minimum:
                        minimum = start
        self.results["maximum money"].append(maximum)
        self.results["% successful trades"].append(0)
        self.results["average income per trade"].append(0)
        self.results["minimum money"].append(minimum)
        self.results["money at the end"].append(start)
        self.results["buy operations"].append(buy)
        self.results["sell operations"].append(sell)
        # print(f"Strategy: {actual}")
        # print(f"Money at the end: {start}")
        # print("---------------------------------")
        return self.results
