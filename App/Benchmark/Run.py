import numpy as np
from numpy import genfromtxt
from Fibonacci import Fibonacci as Fib
from Trade import Trade as T
from Levels import Levels as L
from Trend import Trend as TR
from Buy_And_Sell import Buy_And_Sell as BAS

#my_data1 = genfromtxt("EURUSD_Candlestick_1_M_BID_01.09.2018-30.09.2018.csv", delimiter=',')
#my_data2 = genfromtxt("EURUSD_Candlestick_1_M_BID_01.10.2018-31.10.2018.csv", delimiter=',')
my_data = genfromtxt("EURUSD_Candlestick_1_M_BID_01.10.2017-30.09.2018.csv", delimiter=',')
#my_data = np.concatenate((my_data1, my_data2))
trend_array = []
level = None
balance_EUR = 10000
balance_USD = 0
trade_values = None
count = 0
buy_sell = None
old = [0] * 2

# 1260 as a window size works well


for i in range(len(my_data)-1260):
    window = my_data[i:1260+i]
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

print(i)
print(balance_EUR, balance_USD, my_data[-1][4])
