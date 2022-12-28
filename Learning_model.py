from Indicators import IndicatorsAnalysis
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing import sequence

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# class containing simulation
class LearningModel:
    def __init__(self, indicator_analysis: IndicatorsAnalysis):
        self.indicator_analysis = indicator_analysis
        self.results = {"Strategy type": [], "% successful trades": [], "average income per trade": [],
                                             "maximum money": [], "minimum money": [], "money at the end": [],
                        "buy operations": [], "sell operations": []}

    def testing(self, given_strategy):
        ...
    #     # Simulation base settings
    #     start = 1000.0
    #     accumulated_crypto = 0.0
    #     average = 0.0
    #     maximum = start
    #     minimum = start
    #     buy = 0
    #     sell = 0
    #     fee = 0.001
    #     actual = list()
    #     for x in given_strategy:
    #         actual.append(x)
    #     self.results['Strategy type'].append(actual)
    #     # Testing combination
    #     for index, row in self.indicator_analysis.sma_result.iterrows():
    #         for z in actual:
    #             if row[str(z)] == "Buy":
    #                 buy = buy + 1
    #                 buy_amount = start * 0.5
    #                 accumulated_crypto = accumulated_crypto + (buy_amount - buy_amount * fee) / row['close']
    #                 start = start - buy_amount
    #             elif row[str(z)] == "Sell":
    #                 sell = sell + 1
    #                 sell_amount = accumulated_crypto * 0.7
    #                 start = start + (sell_amount - sell_amount * fee) * row['close']
    #                 accumulated_crypto = accumulated_crypto - sell_amount
    #                 if start > maximum:
    #                     maximum = start
    #                 elif start < minimum:
    #                     minimum = start
    #             if index == len(self.indicator_analysis.sma_result) - 1:
    #                 start = start + accumulated_crypto * row['close']
    #                 if start > maximum:
    #                     maximum = start
    #                 elif start < minimum:
    #                     minimum = start
    #     # Saving results to a DataFrame
    #     self.results["maximum money"].append(maximum)
    #     self.results["% successful trades"].append(0)
    #     self.results["average income per trade"].append(0)
    #     self.results["minimum money"].append(minimum)
    #     self.results["money at the end"].append(start)
    #     self.results["buy operations"].append(buy)
    #     self.results["sell operations"].append(sell)
    #     return self.results



class RegresjaLogistyczna:
    def __init__(self, data):
        self.data = data.replace(["Buy", "Sell", None], [1, -1, 0])
        self.data["zysk_1h"] = self.zysk_1h()
        self.data["zysk_4h"] = self.zysk_4h()
        self.data["zysk_8h"] = self.zysk_8h()
        self.data_wag = self.wagi(self.data)
        self.model = self.model()

    def wagi(self, data):
        for index, i in data.iterrows():
            if data["crossovers"][index] == 1 and data["bb_squezze"][index] == 1:
                data["crossovers"][index] = 6
                data["bb_squezze"][index] = 4
            elif data["crossovers"][index] == 1:
                data["crossovers"][index] = 6
            elif data["crossovers"][index] == 1 and data["bb_squezze"][index] == 1 and data["cci_100_bb_cross"][index] == 1:
                data["crossovers"][index] = 6
                data["bb_squezze"][index] = 4
                data["cci_100_bb_cross"][index] = 3
            elif data["crossovers"][index] == 1 and data["cci_100_bb_cross"][index] == 1:
                data["crossovers"][index] = 6
                data["cci_100_bb_cross"][index] = 3
            elif data["crossovers"][index] == 1 and data["bb_squezze"][index] == 1 and data["50_to_100"][index] == 1:
                data["crossovers"][index] = 6
                data["bb_squezze"][index] = 4
                data["50_to_100"][index] = 2
            elif data["crossovers"][index] == -1 and data["cci_100_bb_cross"][index] == -1:
                data["crossovers"][index] = -6
                data["cci_100_bb_cross"][index] = -3
            elif data["crossovers"][index] == -1:
                data["crossovers"][index] = -6
            elif data["crossovers"][index] == -1 and data["bb_squezze"][index] == -1 and data["cci_100_bb_cross"][index] == -1:
                data["crossovers"][index] = -6
                data["bb_squezze"][index] = -4
                data["cci_100_bb_cross"][index] = -3
            elif data["crossovers"][index] == -1 and data["cci_100_bb_cross"][index] == -1:
                data["crossovers"][index] = -6
                data["cci_100_bb_cross"][index] = -3
            elif data["crossovers"][index] == -1 and data["bb_squezze"][index] == -1 and data["50_to_100"][index] == -1:
                data["crossovers"][index] = -6
                data["bb_squezze"][index] = -4
                data["50_to_100"][index] = -2
        return data

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def zysk_1h(self):
        zysk = list()
        for index, i in self.data.iterrows():
            if index >= len(self.data['close'])-1:
                zysk.append(0)
            else:
                wynik = (self.data['close'][index + 1] - self.data['close'][index])
                if wynik > 0:
                    zysk.append(1)
                else:
                    zysk.append(0)

        return zysk

    def zysk_4h(self):
        zysk = list()
        for index, i in self.data.iterrows():
            if index >= len(self.data['close'])-4:
                zysk.append(0)
            else:
                wynik = (self.data['close'][index + 4] - self.data['close'][index])
                if wynik > 0:
                    zysk.append(1)
                else:
                    zysk.append(0)
        return zysk

    def zysk_8h(self):
        zysk = list()
        for index, i in self.data.iterrows():
            if index >= len(self.data['close'])-8:
                zysk.append(0)
            else:
                wynik = (self.data['close'][index + 8] - self.data['close'][index])
                if wynik > 0:
                    zysk.append(1)
                else:
                    zysk.append(0)
        return zysk

    def model(self):
        dataa = self.data_wag.drop(["open_time", "close", "open", "sma_20_prev", "sma_50_prev", "sma_100_prev",
                                "close_prev", "bb_up_prev", "bb_down_prev", "rsi_prev"], axis=1)
        dataaa = dataa.loc[(dataa.crossovers != 0) | (dataa["50_to_100"] != 0) | (dataa.sma_cross_rsi != 0) |
                       (dataa.bb_squezze != 0) |(dataa.cci_100_bb_cross != 0) | (dataa.rsi_chart != 0) |
                       (dataa.rsi_bb_boundary != 0)]
        data_1h = dataaa.iloc[:, :7]
        target = dataaa.iloc[:, -1]
        # a = list()
        X_train, X_test, Y_train, Y_test = train_test_split(data_1h, target, test_size=0.30)
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, Y_train)
        y_pred = log_reg.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred)
        plot_confusion_matrix(cm)
        print(accuracy_score(Y_test, y_pred))

        # Create the neural network model
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=10000)

        # Train the model on the data
        model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        ...




















            #print(f'Accuracy: {accuracy_score(Y_test, y_pred)}')
        # y_pred = log_reg.predict(X_test)
        # cm = confusion_matrix(Y_test, y_pred)
        # plot_confusion_matrix(cm)
        # print(f'Accuracy: {accuracy_score(Y_test, y_pred)}')
        # y_pred = log_reg.predict(X_test)
        # cm = confusion_matrix(Y_test, y_pred)
        # plot_confusion_matrix(cm)
        # print(r2_score(Y_test, y_pred)))

        # regressor = LinearRegression()
        # regressor.fit(data_1h, target)
        # y_pred_lin = regressor.predict(data_1h)

        # plt.figure(figsize=(8, 6))
        # plt.title('Regresja wielomianowa')
        # plt.xlabel('cecha x')
        # plt.ylabel('zmienna docelowa')
        # plt.scatter(data_1h.crossovers, target, label='cecha x')
        # plt.plot(data_1h.crossovers, y_pred_lin, c='red', label='regresja liniowa')
        # plt.legend()
        # plt.show()

        # self.data.crossovers.replace([1, -1], ["crossovers_up", "crossovers_down", 0])


        # fix random seed for reproducibility
        # tf.random.set_seed(7)
        # embedding_vecor_length = 32
        # model = Sequential()
        # model.add(Embedding(50, embedding_vecor_length, input_length=7))
        # model.add(LSTM(100))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        # model.fit(X_train, Y_train, epochs=50, batch_size=64)
        # # Final evaluation of the model
        # scores = model.evaluate(X_test, Y_test, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1] * 100))


        ...

