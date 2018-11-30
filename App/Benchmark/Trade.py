class Trade:
    def __init__(self, value, balance_EUR, balance_USD, bound):
        self.spread = value[5] - value[1]
        self.value = value[1]
        self.limit = 0
        self.balance_EUR = balance_EUR
        self.balance_USD = balance_USD
        self.bound = bound
        self.limit = self.buy_or_sell()

    def buy_or_sell(self):
        if self.bound == 0 and self.balance_USD > 0:
            return self.value - 0.00002
        elif self.bound == 1 and self.balance_EUR > 0:
            print(self.value)
            print(self.value + self.spread + 0.00002)
            return self.value + self.spread + 0.00002
        else:
            return 0
