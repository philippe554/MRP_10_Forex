class Levels:
    def __init__(self, F, value):
        self.F = F
        self.value = value
        self.lower, self.upper, self.L1, self.L2 = self.detect_levels()
        self.dist_lower, self.dist_upper = self.get_dist()
        self.under_lower = False
        self.over_upper = False

    def detect_levels(self):
        v = self.value[4]

        if self.F.low.min() < v <= self.F.fib_23_6:
            return self.F.low.min(), self.F.fib_23_6, 0, 23.6
        elif self.F.fib_23_6 < v <= self.F.fib_38_2:
            return self.F.fib_23_6, self.F.fib_38_2, 23.6, 38.2
        elif self.F.fib_38_2 < v <= self.F.fib_50_0:
            return self.F.fib_38_2, self.F.fib_50_0, 38.2, 50.0
        elif self.F.fib_50_0 < v <= self.F.fib_61_8:
            return self.F.fib_50_0, self.F.fib_61_8, 50.0, 61.8
        elif self.F.fib_61_8 < v <= self.F.fib_76_4:
            return self.F.fib_61_8, self.F.fib_76_4, 61.8, 76.4
        else:
            return self.F.fib_76_4, self.F.high.max(), 76.4, 100

    def get_dist(self):
        return self.value[4] - self.lower, self.upper - self.value[4]

