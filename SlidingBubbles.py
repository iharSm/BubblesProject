import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import datetime as dt
from Bubbles import Bubbles
from statsmodels.formula.api import ols


class SlidingBubbles(Bubbles):
    def __init__(self, path, window=256, step=30, min_date='01/01/1900', max_date='01/01/2100'):
        Bubbles.__init__(self, path)
        self.stock = self.stock[(self.stock.index >= min_date) & (self.stock.index <= max_date)]
        self.window = window
        self.step = step
        self.count = 0

    def min_function(self, contrast):
        min = 0
        max = len(self.stock.index) - 1
        upper = min + self.window
        vol = np.empty((max - self.window) // self.step + 2, dtype=OptimizeResult)
        self.date = np.empty((max - self.window) // self.step + 2, dtype=dt.datetime)

        i = 0
        while upper < max:
            self.date[i] = self.stock.index[upper]
            self.stock_window = self.stock.ix[min:upper, :]
            vol[i] = minimize(contrast, x0=[4, 1], method='L-BFGS-B')
            min += self.step
            upper += self.step
            i += 1

        self.date[i] = self.stock.index[max]
        self.stock_window = self.stock.ix[min:max, :]
        vol[i] = minimize(contrast, x0=[4, 1], method='L-BFGS-B')
        return vol

    def contrast(self, par):
        sigma_0, alpha = par
        x = self.get_power_parametric_vol(sigma_0, alpha, self.stock_window.shift(1))
        G = self.S(self.stock_window)
        n = G.shape[0]
        ss = np.matrix(x ** 2 - G ** 2).reshape(n, 1)
        return (np.dot(ss.T, ss) / n)[0, 0]

    def plot_sigma_time(self, opt, title):
        alpha = [o.x[1] for o in opt]

        host = host_subplot(111)
        par1 = host.twinx()

        plt.title(title)
        host.set_ylabel("alpha")
        par1.set_ylabel("Stock Price")
        par1.plot(self.stock)
        host.plot(self.date, alpha)
        plt.axhline(y=1, color='r', ls='-')

        plt.draw()

        df_alpha = pd.DataFrame(index = self.date, data=alpha)

        df = self.stock.join(df_alpha, how='outer')
        df.columns = ['Stock', 'alpha']
        df["isBubble?"] = df['alpha'] > 1
        df.to_csv("output/" + title + ".csv")
        df["category"] = df["isBubble?"].astype('category')
        model = ols("Stock ~ category", df).fit()
        print(model.summary())

    def analize(self, optimum):
        alpha = [o.x[1] for o in optimum]
        sigma_o = [o.x[0] for o in optimum]
        bubble_alpha = [x for x in alpha if x > 1]
        print('\n optimized values:')
        print('maximum \n    alpha: ', np.max(alpha),
              '\n    corresponding sigma0 value: ', sigma_o[np.argmax(alpha)],
              '\n    date: ', self.date[np.argmax(alpha)])
        print('\n minimum \n    alpha: ', np.min(alpha),
              '\n    corresponding sigma0 value: ', sigma_o[np.argmin(alpha)],
              '\n    date: ', self.date[np.argmin(alpha)])
        print('\nAlpha greater than 1\n    Number of observations:  ', len(alpha),
              '\n    Number of days above 1:  ', len(bubble_alpha),
              '\n    % of total:   ', len(bubble_alpha) / len(alpha))


def run_all(path, window=256, step=30, min_date='01/01/1900', max_date='01/01/2100', plot_stock=False):
    print("\n\n--------------------", path, "-----------------------------\n")
    print('window:  ', window)
    print('step:  ', step)
    print('start date:  ', min_date)
    print('end date:  ', max_date)

    bubbles = SlidingBubbles(path=path, window=window, step=step, min_date=min_date, max_date=max_date)
    optimum = bubbles.min_function(contrast=bubbles.contrast)
    bubbles.analize(optimum)
    title = path.split("/")[-1] + "_" + str(window) + "_" + str(step)
    bubbles.plot_sigma_time(optimum, title=title)
    print("\n------------------------------END-------------------------------------------")


#
# Here is how to use this code
#
# You don't need to change or edit anython above, just the code below!
#
# to run your code call the function run_all with the following parameters
#
# 1. this is a path to a csv data file. It has to be either relative to the folder that you place
# this SlidingBubbles.py file (expample below: "data/new_data/BoAFixed_FloatABSIndex.csv) or the full path (C:/blabla/blabla.csv)
#
# 2. window. this is number of data points that will be used to fit the sigma function.
# 3. step. this is by how much window will be shifted on each iteration
# 4. min_date and max_date. these are the dates that you want to limit your analysis to.
# 5. plot_stock. This is just a parameter that tells wether you want to plot the stock chart or not. By default it is set to no.
#
# to recap
#
# first iteration:
# min_date                                                         max_date
#     |--------------------------------------------------------------|
#     |-----window=256-----|
#
# second iteration:
# min_date                                                         max_date
#     |--------------------------------------------------------------|
#     |-step=30--||-----window=256-----|
#
# third iteration:
# min_date                                                         max_date
#     |--------------------------------------------------------------|
#     |-step=30--||-step=30--||-----window=256-----|
#

run_all("data/new_data/BoAFixed_FloatABSIndex.csv", window=256, step=30,  min_date='01/01/1900', max_date='01/01/2100', plot_stock=True)


plt.show()
