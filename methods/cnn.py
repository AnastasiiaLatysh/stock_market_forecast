import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D
from numpy import array
import numpy as np
from methods.methods import Methods, DATA_FILE


class ConvolutionalNeuralNetwork(Methods):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        self.data_of_shares = pd.read_csv(DATA_FILE)
        self.data_of_shares['Date'] = pd.to_datetime(self.data_of_shares.Date)
        self.data_of_shares.sort_values(by='Date', ascending=False, inplace=True)
        self.data_of_shares["average"] = (self.data_of_shares["High"] + self.data_of_shares["Low"]) / 2
        self.data_of_shares.index = pd.to_datetime(self.data_of_shares.Date, format='%m/%d/%y')
        self.data_of_shares.head()

        m = self.data_of_shares.average.mean()
        s = self.data_of_shares.average.std()
        price = []
        for x in self.data_of_shares.average:
            price.append((x - m) / s)
        self.data_of_shares.average = price

        self.amount_of_test_data = int(0.2 * len(self.data_of_shares))

    # transform list into supervised learning format
    def series_to_supervised(self, data, n_in=20, n_out=1):
        df = pd.DataFrame(data)
        cols = list()
        # input sequence (t­n, ... t­1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg.values

    # fit a model
    def model_fit(self, train, config, opt, act):
        n_input, n_filters, n_kernel, n_epochs, n_batch = config  # prepare data
        data = self.series_to_supervised(train, n_in=n_input)
        train_x, train_y = data[:, :-1], data[:, -1]
        train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
        model = Sequential()
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation=act, input_shape=(n_input, 1)))
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation=act))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error'])
        model.summary()
        # fit
        h = model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=2)
        return model, h

    # forecast with a pre­fit model
    def model_predict(self, model, history, config):
        # unpack config
        n_input, _, _, _, _ = config
        # prepare data
        x_input = array(history[-n_input:]).reshape((1, n_input, 1))  # forecast
        predicted_value = model.predict(x_input, verbose=2)
        return predicted_value[0]

    # walk­forward validation for univariate data
    def walk_forward_validation(self, data, n_train, cfg, opt, act):
        predictions = list()
        train, test = data[:-self.amount_of_test_data], data[-self.amount_of_test_data:]

        # fit model
        model, h = self.model_fit(train, cfg, opt, act)
        print(f"MSE:{h.history['mean_squared_error']}")
        print(f"MAE: {h.history['mean_absolute_error']}")

        # seed history with training dataset
        history = [x for x in train]
        # plot metrics
        # step over each time­step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.model_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])

        print('TEST MSE:', mean_squared_error(test, predictions))
        print('TEST MAE:', mean_absolute_error(test, predictions))

        predictions = np.concatenate(predictions, axis=0)
        self.create_fig()
        plt.plot(abs(test - predictions), label='Residual for test')
        plt.legend()
        plt.savefig("static/results/cnn_residual")
        return predictions, train

    def start(self):

        config = [20, 256, 5, 200, 32]
        result, tr = self.walk_forward_validation(self.data_of_shares.average[::-1], self.amount_of_test_data, config, 'sgd', 'relu')

        self.data_of_shares['cnn'] = tr
        self.data_of_shares.loc[:self.amount_of_test_data, 'cnn'] = result[::-1]
        self.create_fig()
        plt.plot(self.data_of_shares.average, color='green', label='test')
        plt.plot(self.data_of_shares.average[:self.amount_of_test_data:-1], color='red', label='train')
        plt.plot(self.data_of_shares.cnn[self.amount_of_test_data::-1], color='yellow', label='predicted')
        plt.title("Price of stocks sold")
        plt.xlabel("Time check")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.savefig("static/results/cnn_forecast")
