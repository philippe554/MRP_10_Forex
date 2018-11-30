import numpy as np


class Fibonacci:
    def __init__(self, window):
        self.window = window
        self.high, self.low, = self.get_high_low()
        self.fib_23_6 = 0
        self.fib_38_2 = 0
        self.fib_50_0 = 0
        self.fib_61_8 = 0
        self.fib_76_4 = 0
        self.r_levels()

    def get_high_low(self):
        my_data = np.delete(self.window, 0, 0)
        my_data = np.delete(my_data, 0, 1)
        my_data = my_data[:, [0, 3]]
        return my_data.max(axis=0), my_data.min(axis=0)

    def r_levels(self):
        dif = self.high - self.low
        dif = dif.max()
        self.fib_23_6 = self.low.min() + dif * 0.236
        self.fib_38_2 = self.low.min() + dif * 0.382
        self.fib_50_0 = self.low.min() + dif * 0.5
        self.fib_61_8 = self.low.min() + dif * 0.618
        self.fib_76_4 = self.low.min() + dif * 0.764
