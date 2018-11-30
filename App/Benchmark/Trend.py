class Trend:
    def __init__(self, F, L1, L2):
        self.F = F
        self.L1 = L1
        self.L2 = L2
        self.update_lower, self.update_upper = self.get_levels()
        self.under_thresh = False
        self.above_thresh = False
        self.threshold = 0.0015

    def get_levels(self):
        if self.L1 == 0 and self.L2 == 23.6:
            return self.F.low,  self.F.fib_23_6
        elif self.L1 == 23.6 and self.L2 == 38.2:
            return self.F.fib_23_6, self.F.fib_38_2
        elif self.L1 == 38.2 and self.L2 == 50:
            return self.F.fib_38_2 , self.F.fib_50_0
        elif self.L1 == 50 and self.L2 == 61.8:
            return self.F.fib_50_0, self.F.fib_61_8
        elif self.L1 == 61.8 and self.L2 == 76.4:
            return self.F.fib_61_8, self.F.fib_76_4
        else:
            return self.F.fib_76_4, self.F.high

    def check_thresh(self, value):
        diff_with_upper = self.update_upper - value[1]
        diff_with_lower = value[1] - self.update_lower
        min_diff = min(diff_with_lower, diff_with_upper)

        if 0 < min_diff < self.threshold:
            if min_diff is diff_with_upper:
                result = 1, 1
            else:
                result = 1, 0
        elif min_diff is diff_with_upper:
            result = 0, 1
        else:
            result = 0, 0

        if value[1] - self.update_lower >= self.threshold and self.update_upper - \
                value[1] >= self.threshold:
            result = 2, 3

        return result

