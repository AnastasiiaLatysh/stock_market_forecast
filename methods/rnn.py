import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense, LSTM
from methods.methods import Methods, DATA_FILE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


class RecurrentNeuralNetwork(Methods):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        stock_data = pd.read_csv(DATA_FILE)
        stock_data['Date'] = pd.to_datetime(stock_data.Date)
        stock_data.sort_values(by='Date', ascending=False, inplace=True)

        stock_data["average"] = (stock_data["High"] + stock_data["Low"])/2
        stock_data.index = pd.to_datetime(stock_data.Date,format='%m/%d/%y')
        stock_data.head()

        stock_data.describe()
        input_feature= stock_data.iloc[:,[2,6]].values
        self.input_data=input_feature

        sc= MinMaxScaler(feature_range=(0,1))
        self.input_data[:,0:2] = sc.fit_transform(input_feature[:,:])

        lookback= 20

        self.test_size=int(.3 * len(stock_data))
        self.X=[]
        self.y=[]
        for i in range(len(stock_data)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(self.input_data[[(i + j)], :])
            self.X.append(t)
            self.y.append(self.input_data[i + lookback, 1])
        self.X, self.y= np.array(self.X), np.array(self.y)

        self.X_test = self.X[:self.test_size]
        self.y_test = self.y[:self.test_size]
        self.X_train = self.X[self.test_size:]
        self.y_train = self.y[self.test_size:]
        # X_train.shape, X_test.shape


        self.X = self.X.reshape(self.X.shape[0],lookback, 2)
        self.X_test = self.X_test.reshape(self.X_test.shape[0],lookback, 2)
        self.X_train = self.X_train.reshape(self.X_train.shape[0],lookback, 2)

        print(self.X.shape)
        print(self.X_test.shape)
        print(self.X_train.shape)

    def start(self):
        model = Sequential()
        model.add(LSTM(units=30, return_sequences= True, input_shape=(self.X.shape[1],2), activation="relu"))
        model.add(LSTM(units=30, return_sequences=True, activation="relu"))
        model.add(LSTM(units=30, activation="relu"))
        model.add(Dense(units=1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
        model.fit(self.X_train, self.y_train, epochs=100, batch_size=32)

        predicted_value_test = model.predict(self.X_test)
        predicted_value_train = model.predict(self.X_train)

        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        print('TEST LOSS:', score[0])
        print('TEST MSE:', mean_squared_error(self.y_test, predicted_value_test))
        print('TEST MAE:', mean_absolute_error(self.y_test, predicted_value_test))

        self.create_fig()
        plt.plot(self.input_data[:, 1][::-1], color='green', label='test')
        plt.plot(np.concatenate((predicted_value_test, predicted_value_train))[::-1], color='yellow', label='predicted')

        plt.plot(np.concatenate((predicted_value_test, predicted_value_train))[:self.test_size:-1], color='red', label='train')
        plt.title("Price of stocks sold")
        plt.xlabel("Time check")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.savefig("static/results/rnn_forecast")
