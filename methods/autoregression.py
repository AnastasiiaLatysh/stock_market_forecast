from methods.methods import Methods
import statsmodels.api as sm

import matplotlib.pyplot as plt


class AutoregressionMethods(Methods):

    def __init__(self):
        super().__init__()
        self.y = self.data_of_shares.Average[::-1]

    def ar(self):
            print("AR!!!")
            mod = sm.tsa.statespace.SARIMAX(self.y, order=(2, 0, 0), seasonal_order=(0, 0, 0, 12),
                                            enforce_stationarity=False, enforce_invertibility=False)
            results_ar = mod.fit()

            pred_ar = results_ar.get_prediction(start=len(self.train_data), dynamic=False)
            print(results_ar.summary().tables[1])

            self.create_fig()
            ax = self.test_data.Average.plot(label='test', color='blue')
            pred_ar.predicted_mean.plot(ax=ax, label='AR(3)', alpha=.7, color='red')
            ax.set_xlabel('Time')
            ax.set_ylabel('Share price')
            plt.legend()
            plt.savefig("static/results/ar_forecast")

            self.create_fig()
            y_forecasted = pred_ar.predicted_mean
            mse = ((y_forecasted - self.test_data.Average) ** 2).mean()
            mae = (abs(y_forecasted - self.test_data.Average)).mean()
            print('AR MSE {}'.format(round(mse, 6)))
            print('AR MAE {}'.format(round(mae, 6)))
            residual_ar = y_forecasted - self.test_data.Average
            plt.plot(residual_ar, label="Residual")
            plt.savefig("static/results/ar_residual")

    def arma(self):
        print("ARMA!!!")
        mod = sm.tsa.statespace.SARIMAX(self.y, order=(3, 0, 1),
                                        seasonal_order=(0, 0, 0, 12), enforce_stationarity=False,
                                        enforce_invertibility=False)
        results_arma = mod.fit()
        print(results_arma.summary().tables[1])
        pred_arma = results_arma.get_prediction(start=len(self.train_data), dynamic=False)
        self.create_fig()
        ax = self.test_data.Average.plot(label='test', color='blue')
        pred_arma.predicted_mean.plot(ax=ax, label='ARMA(3,1)', alpha=.7, color='red')

        ax.set_xlabel('Time')
        ax.set_ylabel('Share price')
        plt.legend()
        plt.savefig("static/results/arma_forecast")

        y_forecasted = pred_arma.predicted_mean
        self.create_fig()
        mse = ((y_forecasted - self.test_data.Average) ** 2).mean()
        mae = (abs(y_forecasted - self.test_data.Average)).mean()
        print('ARMA MSE {}'.format(round(mse, 6)))

        print('ARMA MAE {}'.format(round(mae, 6)))
        residual_arma = y_forecasted - self.test_data.Average
        plt.plot(residual_arma, label="Residual")
        plt.savefig("static/results/arma_residual")

    def arima(self):
        print("ARIMA!!!")
        mod = sm.tsa.statespace.SARIMAX(self.y, order=(2, 1, 1), seasonal_order=(0, 0, 0, 12),
                                        enforce_stationarity=False, enforce_invertibility=False)
        results_arima = mod.fit()

        print(results_arima.summary().tables[1])
        pred_arima = results_arima.get_prediction(start=len(self.train_data), dynamic=False)
        self.create_fig()
        ax = self.test_data.Average.plot(label='test', color='blue')
        pred_arima.predicted_mean.plot(ax=ax, label='ARIMA(2,1,1)', alpha=.7, color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Share price')
        plt.legend()
        plt.savefig("static/results/arima_forecast")

        self.create_fig()
        y_forecasted = pred_arima.predicted_mean
        mse = ((y_forecasted - self.test_data.Average) ** 2).mean()
        mae = (abs(y_forecasted - self.test_data.Average)).mean()
        print('ARIMA MSE {}'.format(round(mse, 6)))
        print('ARIMA MAE {}'.format(round(mae, 6)))
        residual_arima = y_forecasted - self.test_data.Average
        plt.plot(residual_arima, label="Residual")
        plt.savefig("static/results/arima_residual")
