class Levels:
    def __init__(self, F, value):
        self.F = F
        self.value = value
        if self.F.d_trend:
            self.lower, self.upper, self.L1, self.L2 = self.detect_levels_down()
        else:
            self.lower, self.upper, self.L1, self.L2 = self.detect_levels_up()
        self.dist_lower, self.dist_upper = self.get_dist()
        self.under_lower = False
        self.over_upper = False

    def detect_levels_down(self):
        v = self.value[1]
        if self.F.low < v <= self.F.fib_23_6:
            return self.F.low, self.F.fib_23_6, 0, 23.6
        elif self.F.fib_23_6 < v <= self.F.fib_38_2:
            return self.F.fib_23_6, self.F.fib_38_2, 23.6, 38.2
        elif self.F.fib_38_2 < v <= self.F.fib_50_0:
            return self.F.fib_38_2, self.F.fib_50_0, 38.2, 50.0
        elif self.F.fib_50_0 < v <= self.F.fib_61_8:
            return self.F.fib_50_0, self.F.fib_61_8, 50.0, 61.8
        elif self.F.fib_61_8 < v <= self.F.fib_76_4:
            return self.F.fib_61_8, self.F.fib_76_4, 61.8, 76.4
        else:
            return self.F.fib_76_4, self.F.high, 76.4, 100

    def detect_levels_up(self):
        v = self.value[4]

        if self.F.low.min() < v <= self.F.fib_76_4:
            return self.F.low.min(), self.F.fib_76_4, 100, 76.4
        elif self.F.fib_76_4 < v <= self.F.fib_61_8:
            return self.F.fib_76_4, self.F.fib_61_8, 76.4, 61.8
        elif self.F.fib_61_8 < v <= self.F.fib_50_0:
            return self.F.fib_61_8, self.F.fib_50_0, 61.8, 50.0
        elif self.F.fib_50_0 < v <= self.F.fib_38_2:
            return self.F.fib_50_0, self.F.fib_38_2, 50.0, 38.2
        elif self.F.fib_38_2 < v <= self.F.fib_23_6:
            return self.F.fib_38_2, self.F.fib_23_6, 38.2, 23.6
        else:
            return self.F.fib_23_6, self.F.high.max(), 23.6, 0

    def get_dist(self):
        return self.value[1] - self.lower, self.upper - self.value[1]

