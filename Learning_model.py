from Indicators import IndicatorsAnalysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.neural_network import MLPRegressor



# class containing simulation
class LearningModel:
    def __init__(self, indicator_analysis: IndicatorsAnalysis):
        self.indicator_analysis = indicator_analysis
        self.results = {"Strategy type": [], "% successful trades": [], "average income per trade": [],
                                             "maximum money": [], "minimum money": [], "money at the end": [],
                        "buy operations": [], "sell operations": []}


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
        X_train, X_test, Y_train, Y_test = train_test_split(data_1h, target, test_size=0.30)
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




