import numpy as np
import threading

from numpy import genfromtxt
from App.Benchmark.Fibonacci import Fibonacci as Fib
from App.Benchmark.Trade import Trade as T
from App.Benchmark.Levels import Levels as L
from App.Benchmark.Trend import Trend as TR
from App.Benchmark.Buy_And_Sell import Buy_And_Sell as BAS
from App.Library.Connection.Price import Price


tren_array = []
levels = None
bal_EUR = 10000
bal_USD = 0
trad_values = None
c = 0
buy_s = None
o = [0] * 2
i2 = 0
b = -50



# 1260 as a window size works well


def run(count, trade_values, buy_sell, level, i, trend_array, balance_EUR, balance_USD, old, bound):

    candles = np.array(Price.get_last_n_candles(instrument='EUR/USD', period='m1', n=1260))
    window = candles
    fibonacci = Fib(window)

    if count == 5:
        trade_values.limit = 0
        count = 0

    if trade_values is not None and buy_sell is not None:
        balance_EUR, balance_USD, activated = buy_sell.activate_stop_loss(window[-1][4])

        if activated:
            print("old stoploss activated")
            print("Balance at iteration --> ", i, " Euro -> ", balance_EUR, " USD -> ", balance_USD, " value -> ", window[-1][4])
            buy_sell.stop_loss = [0] * 2
            level = L(fibonacci, window[-1])
            trade_values.limit = 0

    if trade_values and trade_values.limit != 0:
        if buy_sell is not None:
            old = buy_sell.stop_loss

        buy_sell = BAS(window[-1], trade_values.limit, balance_EUR, balance_USD, old)

        if balance_EUR == buy_sell.balance_EUR:
            count = count + 1
            trend_array = []
        else:
            balance_EUR = buy_sell.balance_EUR
            balance_USD = buy_sell.balance_USD

            print("Balance at iteration --> ", i, " Euro -> ", balance_EUR, " USD -> ", balance_USD, " value -> ", window[-1][4])
            count = 0
            trade_values.limit = 0
            trend_array = []

    if i > 0:
        trend = TR(fibonacci, level.L1, level.L2)
        level.lower = trend.update_lower
        level.upper = trend.update_upper
        check, bound = trend.check_thresh(window[-1])
        if check == 2:
            trend_array = []
        else:
            trend_array.append(check)

    if i is 0:
        level = L(fibonacci, window[-1])

    if len(trend_array) >= 7:
        if sum(trend_array[:7]) >= 4:
            trade_values = T(window[-1], balance_EUR, balance_USD, bound)
            trend_array = []
        elif sum(trend_array[:7]) < 4 and buy_sell is not None and buy_sell.stop_loss[0] == 0:
            level = L(fibonacci, window[-1])
            trend_array = []
        elif sum(trend_array[:7]) < 4:
            trend_array = []

    i = i+1
    print(i)
    print(balance_EUR, balance_USD, candles[-1][1])
    print(len(trend_array))
    return count, trade_values, buy_sell, level, i, trend_array, balance_EUR, balance_USD, old, bound


import time

while True:
    c, trad_values, buy_s, levels, i2, tren_array, bal_EUR, bal_USD, o, b = run(c, trad_values, buy_s, levels, i2,
                                                                                tren_array, bal_EUR, bal_USD, o, b)
    time.sleep(60)



# print(i2)
# print(balance_EUR, balance_USD, candles[-1][4])
