import os

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
DATA_FILE = f'{ROOT_DIR}/app/data/dataset.csv'


class Methods(object):

    def __init__(self):
        self.data_of_shares = None
        self.test_data = None
        self.train_data = None
        self.amount_of_test_data = None
        self.prepare_data()

    def prepare_data(self):
        # 1. Load data (Завантажуємо дані з файлу)
        self.data_of_shares = pd.read_csv(DATA_FILE)
        self.data_of_shares['Date'] = pd.to_datetime(self.data_of_shares.Date)
        self.data_of_shares.sort_values(by='Date', ascending=False, inplace=True)

        # 2. Calculate "average" and put it into table as additional column
        # (Розраховуємо середнє значення акцій і вставляємо їх як додаткову колонку)
        self.data_of_shares["Average"] = (
            self.data_of_shares["High"] + self.data_of_shares["Low"]) / 2
        print(f"Current DataSet are\n: {self.data_of_shares.head()}")

        # 3. Aggregating the dataset at daily level (Агрегування набору даних на
        # щоденному рівні)
        # self.data_of_shares.index = pd.DatetimeIndex(self.data_of_shares.index).to_period('M')
        self.data_of_shares.index = pd.to_datetime(
            self.data_of_shares['Date'], format='%m/%d/%y')
        # self.data_of_shares.index = pd.DatetimeIndex(self.data_of_shares.Date).to_period('M')

        # 4. Scale "average" column in range from 0 to 1 (Масштабування середніх
        # значень цін акцій у проміжку від 0 до 1)
        x = self.data_of_shares['Average'].values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
        self.data_of_shares['Average'] = x_scaled

        # 5. Create train and test set
        # (Створення навчальної та тестової вибірок, де 20 % взідних данних - тестова вибірка, 80 % - навчальна)
        self.amount_of_test_data = int(0.2 * len(self.data_of_shares))
        self.test_data = self.data_of_shares[0:self.amount_of_test_data]
        self.train_data = self.data_of_shares[self.amount_of_test_data:]

    def create_fig(self):
        plt.cla()
        plt.figure(figsize=(16, 8))
        plt.legend()
        plt.xlabel("Time")

    def create_fig_with_data(self):
        self.create_fig()
        plt.plot(self.train_data['Average'], label='Train,', color='green')
        plt.plot(self.test_data['Average'], label='Test', color='red')
