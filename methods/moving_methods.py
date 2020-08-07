import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

from methods.methods import Methods

matplotlib.use('Agg')


class MovingMethods(Methods):

    WINDOWS = [30, 10, 5]
    PARAMETERS = [12, 1, 0.5]

    def moving_average_forecast(self):
        self.create_fig_with_data()

        # Moving average method (метод ковзного середнього)
        for w in self.WINDOWS:
            y_hat = ((self.data_of_shares['Average'].iloc[::-1]
                      ).rolling(w).mean()).iloc[::-1]
            plt.plot(y_hat[0:self.amount_of_test_data],
                     label=f'Moving average forecast window={w}')
        plt.legend(loc='best')
        plt.savefig("static/results/moving_average_forecast")

    def moving_average_errors(self):
        self.create_fig()
        print("Moving Average Method: ")
        for w in self.WINDOWS:
            y_hat = ((self.data_of_shares['Average'].iloc[::-1]
                      ).rolling(w).mean()).iloc[::-1]
            mse = round(mean_squared_error(
                self.test_data['Average'], y_hat[0:self.amount_of_test_data]), 6)
            mae = round(mean_absolute_error(
                self.test_data['Average'], y_hat[0:self.amount_of_test_data]), 6)
            residual = self.test_data['Average'] - y_hat[0:self.amount_of_test_data]
            plt.plot(residual, label=f'Residual for window={w}')
            print(f"MSE fpr window {w}: {mse}")
            print(f"MAE fpr window {w}: {mae}")
        plt.legend(loc='best')
        plt.savefig("static/results/moving_average_errors")

    def weighted_average_forecast(self):
        self.create_fig_with_data()
        for p in self.PARAMETERS:
            y_hat = self.data_of_shares['Average'].ewm(
                halflife=p).mean()
            plt.plot(y_hat[0:self.amount_of_test_data],
                     label='Moving average forecast halflife=' + str(p))
        plt.legend(loc='best')
        plt.savefig("static/results/weighted_average_forecast")

    def weighted_average_errors(self):
        self.create_fig()
        print("Weighted Average Method: ")

        for p in self.PARAMETERS:
            y_hat = self.data_of_shares['Average'].ewm(
                halflife=p).mean()
            mse = mean_squared_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data])
            mae = round(mean_absolute_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            residual = self.test_data.Average - y_hat[0:self.amount_of_test_data]
            plt.plot(residual, label='Residual for parameter=' + str(p))
            print(f"MSE fpr parameter {p}: {mse}")
            print(f"MAE fpr parameter {p}: {mae}")
        plt.legend(loc='best')
        plt.savefig("static/results/weighted_average_errors")

    def exp_moving_average_forecast(self):
        parameters = [0.1, 0.2, 0.6]
        self.create_fig_with_data()
        for p in parameters:
            fit1 = SimpleExpSmoothing(
                self.data_of_shares.Average).fit(
                smoothing_level=p,
                optimized=False)
            y_hat = fit1.fittedvalues
            plt.plot(y_hat[0:self.amount_of_test_data],
                     label=f'Simple exp smoothing level={p}')
        plt.legend(loc='best')
        plt.savefig("static/results/exp_moving_forecast")

    def exp_moving_average_errors(self):
        parameters = [0.1, 0.2, 0.6]
        self.create_fig()
        print("Exponential Average Method: ")
        for p in parameters:
            fit1 = SimpleExpSmoothing(
                self.data_of_shares.Average).fit(
                smoothing_level=p,
                optimized=False)
            y_hat = fit1.fittedvalues
            mse = round(mean_squared_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            mae = round(mean_absolute_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            residual = self.test_data.Average - y_hat[0:self.amount_of_test_data]
            plt.plot(residual, label=f'Residual for parameter={p}')
            print(f"MSE fpr parameter {p}: {mse}")
            print(f"MAE fpr parameter {p}: {mae}")
        plt.legend(loc='best')
        plt.savefig("static/results/exp_moving_errors")

    def double_exp_moving_average_forecast(self):

        parameters = [[0.1, 0.3], [0.2, 0.8], [0.6, 0.6]]
        self.create_fig_with_data()
        for p, s in parameters:
            fit1 = Holt(
                self.data_of_shares.Average).fit(
                smoothing_level=p,
                smoothing_slope=s)
            y_hat = fit1.fittedvalues
            plt.plot(y_hat[0:self.amount_of_test_data],
                     label=f'Double exp smoothing level={p}, slope={s}')
        plt.legend(loc='best')
        plt.savefig("static/results/double_exp_moving_forecast")

    def double_exp_moving_average_errors(self):

        parameters = [[0.1, 0.3], [0.2, 0.8], [0.6, 0.6]]
        self.create_fig()
        print("Double Exponential Average Method: ")
        for p, s in parameters:
            fit1 = Holt(
                self.data_of_shares.Average).fit(
                smoothing_level=p,
                smoothing_slope=s)
            y_hat = fit1.fittedvalues
            mse = round(mean_squared_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            mae = round(mean_absolute_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            residual = self.test_data.Average - y_hat[0:self.amount_of_test_data]
            plt.plot(residual, label=f'Residual for level={p}, slope={s}')
            print(f"MSE for level={p}, slope={s}: {mse}")
            print(f"MAE for level={p}, slope={s}: {mae}")

        plt.legend(loc='best')
        plt.savefig("static/results/double_exp_moving_errors")

    def holt_winters_method(self):
        parameters = [[10, 'add', 'add'], [6, 'add', 'add']]
        self.create_fig_with_data()
        for p, tr, seas in parameters:
            fit1 = ExponentialSmoothing(
                self.data_of_shares.Average,
                seasonal_periods=p,
                trend=tr,
                seasonal=seas).fit()
            y_hat = fit1.fittedvalues
            plt.plot(y_hat[0:self.amount_of_test_data], label=f'Holt­Winters periods={p}, trend={tr} seasonal = {seas}')
        plt.legend(loc='best')
        plt.savefig("static/results/holt_winters_method")

    def holt_winters_method_errors(self):
        parameters = [[10, 'add', 'add'], [6, 'add', 'add']]
        self.create_fig()
        for p, tr, seas in parameters:
            fit1 = ExponentialSmoothing(
                self.data_of_shares.Average,
                seasonal_periods=p,
                trend=tr,
                seasonal=seas).fit()
            y_hat = fit1.fittedvalues
            mse = round(mean_squared_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            mae = round(mean_absolute_error(
                self.test_data.Average, y_hat[0:self.amount_of_test_data]), 6)
            residual = self.test_data.Average - y_hat[0:self.amount_of_test_data]
            plt.plot(residual, label=f'Residual for periods={p}, trend={tr} seasonal = {seas}')
            print(f"MSE for periods={p}, trend={tr} seasonal = {seas}: {mse}")
            print(f"MAE for periods={p}, trend={tr} seasonal = {seas}: {mae}")
        plt.legend(loc='best')
        plt.savefig("static/results/holt_winters_method_errors")
