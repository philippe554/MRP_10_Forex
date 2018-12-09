import numpy as np


class Fibonacci:
    def __init__(self, window):
        self.window = window
        self.high, self.low, self.up, self.down = self.get_high_low()
        self.fib_23_6 = 0
        self.fib_38_2 = 0
        self.fib_50_0 = 0
        self.fib_61_8 = 0
        self.fib_76_4 = 0
        if self.up < self.down:
            self.r_levels_down()
            self.d_trend = True
        elif self.up == self.down:
            self.r_levels_down()
            self.d_trend = True
        else:
            self.r_levels_up()
            self.d_trend = False

    def get_high_low(self):
        my_data = self.window[:, [1]]
        return my_data.max(), my_data.min(), my_data.argmax(), my_data.argmin()

    def r_levels_down(self):
        dif = self.high - self.low
        self.fib_23_6 = self.low.min() + dif * 0.236
        self.fib_38_2 = self.low.min() + dif * 0.382
        self.fib_50_0 = self.low.min() + dif * 0.5
        self.fib_61_8 = self.low.min() + dif * 0.618
        self.fib_76_4 = self.low.min() + dif * 0.764

    def r_levels_up(self):
        dif = self.high - self.low
        self.fib_23_6 = self.high - dif * 0.236
        self.fib_38_2 = self.high - dif * 0.382
        self.fib_50_0 = self.high - dif * 0.5
        self.fib_61_8 = self.high - dif * 0.618
        self.fib_76_4 = self.high - dif * 0.764
