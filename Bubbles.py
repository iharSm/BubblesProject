import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D


class Bubbles:
    def __init__(self, path):
        self.stock = pd.read_csv(filepath_or_buffer=path, index_col=0, parse_dates=True).fillna(method='ffill')

    def get_power_parametric_vol(self, sigma_0, alpha, S):
        return (sigma_0 * (S ** alpha)).dropna()

    # find better name/
    # prices is a dataframe column
    def S(self, prices):
        diff = (prices - prices.shift(1)).dropna()
        n = diff.shape[0]
        return np.sqrt(n) * diff

    def contrast(self, par):
        sigma_0, alpha = par
        x = self.get_power_parametric_vol(sigma_0, alpha, self.stock.shift(1))
        G = self.S(self.stock)
        n = G.shape[0]
        ss = np.matrix(x ** 2 - G ** 2).reshape(n, 1)
        return (np.dot(ss.T, ss) / n)[0, 0]

    def min_function(self, contrast):
        bnds = ((0, None), (0, None))
        return minimize(contrast, x0=[4, 1], method='L-BFGS-B')

    def plot_contrast(self, ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    def plot_stock(self, title):
        # len = self.stock.shape[0]
        # x = np.arange(len)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        self.stock.plot()
        # plt.plot(x,self.stock['Bid'].values)
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        # plt.show()

    def plot_sigma(self, sigma_0, alpha, title):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)

        ax.set_ylabel('Sigma')
        ax.set_xlabel('Stock Price')
        min = 10
        max = 400
        S = np.arange(min, max).reshape(max - min, 1)
        sigma = sigma_0 * np.power(S, alpha)
        ax.plot(S, sigma)


def run_all(path):
    print("\n\n--------------------", path, "-----------------------------\n")
    bubbles = Bubbles(path=path)
    title = path.split("/")[-1].split(".")[0]
    bubbles.plot_stock(title)
    optimum = bubbles.min_function(contrast=bubbles.contrast)
    print(optimum)

    bubbles.plot_sigma(*optimum.x, title)
    print("\n--------------------END-----------------------------------")

# do graphs for last week
#
run_all("data/Data1/Geocities.csv")
run_all("data/Data1/Etoys.csv")
run_all("data/Data1/infospace.csv")
run_all("data/Data1/Linkedin.csv")
run_all("data/Data1/lastminute2.csv")


# run_all("data/new_data/BoAFixed_FloatABSIndex.csv")
# run_all("data/new_data/BoA MBS Index 2006 - 2017.csv")
# run_all("data/new_data/BoA MBS Index 2006 - 2017_2.csv")
# run_all("data/new_data/Brent Crude Oil Futures 2001 - 2017.csv")
# run_all("data/new_data/Dow Jones 2006 - 2017.csv")
# run_all("data/new_data/Gold Futures 2006 - 2017.csv")
# run_all("data/new_data/VIX 2006 - 2017.csv")
run_all("data/new_data/snapchat.csv")
#
plt.show()
