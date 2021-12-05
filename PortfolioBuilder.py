# None of these imports are strictly required, but use of at least some is strongly encouraged
# Other imports which don't require installation can be used without consulting with course staff.
# If you feel these aren't sufficient, and you need other modules which require installation,
# you're welcome to consult with the course staff.


import numpy as np
import pandas as pd
from pandas_datareader import data as web
from datetime import date
from datetime import timedelta
import itertools
import math
import yfinance
from typing import List
import matplotlib.pyplot as plt


class PortfolioBuilder:
    def __init__(self):
        self.time_change = None
        self.x = None
        self.ticker_list = None
        self.data = None
        self.interval = None
        self.end = None
        self.start = None
        self.learn = None

    def get_daily_data(self, tickers_list: List[str], start_date: date, end_date: date = date.today()):
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """
        self.ticker_list = tickers_list
        self.start = start_date
        self.end = end_date
        try:
            data = web.DataReader(self.ticker_list, data_source='yahoo', start=start_date, end=end_date)
        except:
            raise ValueError
        self.data = pd.DataFrame(data)['Adj Close']
        self.time_change = len(self.data)
        return self.data

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        self.interval = portfolio_quantization
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        cover_algo = Cover(self.data, self.ticker_list, self.interval, self.start, self.end)
        return cover_algo.analyze()

    def expo_bt(self):
        time = self.time_change
        b_1 = [1 / len(self.ticker_list)] * len(self.ticker_list)
        bt_list = [np.array(b_1)]
        cover_algo = Cover(self.data, self.ticker_list, self.interval, self.start, self.end)
        self.x = cover_algo.weight()
        d = 0
        if time == 1:
            return bt_list
        else:
            for i in range(1, time):
                n = np.zeros(len(self.ticker_list))
                for j in range(len(self.ticker_list)):
                    n[j] = bt_list[i - 1][j] * math.exp((self.learn * self.x[i - 1][j]) / (bt_list[i - 1] @ self.x[i - 1]))
                    d += n[j]
                bt = n / d
                d = 0
                bt_list.append(bt)
        return bt_list

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
               calculates the exponential gradient portfolio for the previously requested stocks

               :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
               :return: returns a list of floats, representing the growth trading  per day
               """
        self.learn = learn_rate
        W = [1.0]
        imitation = np.zeros(self.time_change - 1)
        for i in range(self.time_change - 1):
            imitation[i] = self.expo_bt()[i] @ self.x[i]
        for j in range(self.time_change - 1):
            s = np.prod(imitation[:j + 1])
            W.append(s)
        return W



class Cover:

    def __init__(self, data, ticker_list, interval, start_date: date, end_date: date = date.today()):
        self.data = data
        self.interval = interval
        self.start = start_date
        self.end = end_date
        self.cov_tik_list = ticker_list
        self.time_change = len(self.data)

    def omega(self):
        table = self.data
        a = self.interval
        n = len(table.columns)
        omega = self.find_all_port(a, n) / a  # vector
        return omega

    def weight(self):
        returns = self.data.pct_change().dropna(how='all', axis=1)

        # Change the data to Numpy for faster calculation
        X = np.array(returns)
        X[np.isnan(X)] = 0

        # add 1 (i.e. change -0.01 -> 0.99)
        X = X + 1.0
        X = X[1:]
        self.weight1 = X
        return self.weight1

    def analyze(self):
        # History of return prices

        # Returns is daily change, remove empty security, remove the last day NaN
        returns = self.data.pct_change().dropna(how='all', axis=1)


        # Change the data to Numpy for faster calculation
        X = np.array(returns)
        X[np.isnan(X)] = 0

        # add 1 (i.e. change -0.01 -> 0.99)
        X = X + 1.0
        X = X[1:]
        self.weight1 = X
        # In theory, we are supposed to calculate the integral of wealth over all portfolio
        # You cannot do that in practice, so we approximate by doing it descreetly.
        B = self.omega()
        bt_list = self.bt_list(X, B)
        return self.find_wealth(bt_list, X)

    def find_wealth(self, bt_list, x_matrix):
        W = [1.0]
        imitation = np.zeros(self.time_change - 1)
        for i in range(self.time_change - 1):
            imitation[i] = bt_list[i] @ x_matrix[i]
        for j in range(self.time_change - 1):
            l = imitation[j]
            w = W[j]
            S = l * w
            W.append(S)
        return W

    def bt_list(self, x_matrix, b_matrix):
        days = self.time_change
        X = x_matrix
        B = b_matrix.tolist()
        bt_list = []
        for i in range(days):
            bank = []
            d = 0
            for b in B:
                st = 1.0
                for t in range(i):
                    st *= np.array(b) @ X[t]
                d += st
                bank.append((st * np.array(b)))
            mass = np.array(bank)
            sum = np.sum(mass, axis=0)
            bw = sum / d
            bt_list.append(bw)
        return bt_list

    def find_all_port(self, n, k, cache={}):
        if n == 0:
            return np.zeros((1, k))
        if k == 0:
            return np.empty((0, 0))
        args = (n, k)
        if args in cache:
            return cache[args]
        a = self.find_all_port(n - 1, k, cache)
        a1 = a + (np.arange(k) == 0)
        b = self.find_all_port(n, k - 1, cache)
        b1 = np.hstack((np.zeros((b.shape[0], 1)), b))
        b1 = np.hstack((np.zeros((b.shape[0], 1)), b))
        result = np.vstack((a1, b1))
        cache[args] = result
        return result


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
    dor = PortfolioBuilder()

    dataTest = dor.get_daily_data(['GOOG', 'AAPL', 'MSFT'], date(2018, 1, 1), date(2020, 5, 14))

    #test_cover = dor.find_universal_portfolio(20)
    #print(test_cover)
    #print(dataTest)
    #print(len(dataTest))
    #print(test_cover[:9])
    #print(len(test_cover))
    print(dor.find_exponential_gradient_portfolio())
    #print(len(dor.find_exponential_gradient_portfolio()))


