class Buy_And_Sell:
    def __init__(self, value, limit, balance_EUR, balance_USD, old):
        self.value = value[4]
        self.limit = limit
        self.balance_USD = balance_USD
        self.balance_EUR = balance_EUR
        self.stop_loss = old
        self.balance_EUR, self.balance_USD, self.stop_loss = self.calculate_new_balance()

    def calculate_new_balance(self):
        old = self.stop_loss
        if self.balance_USD != 0 and self.value <= self.limit:
            return self.balance_USD / self.limit, 0, [self.value - 0.0004, 1]
        elif self.balance_EUR != 0 and self.value >= self.limit:
            return 0, self.balance_EUR * self.limit, [self.value + 0.0004, 0]
        else:
            return self.balance_EUR, self.balance_USD, old

    def activate_stop_loss(self, val):
        if self.stop_loss[1] == 1 and val < self.stop_loss[0] and self.stop_loss[0] > 0:
            self.balance_USD = self.balance_EUR * self.stop_loss[0]
            self.balance_EUR = 0
            return self.balance_EUR, self.balance_USD, True
        elif self.stop_loss[1] == 0 and val > self.stop_loss[0] and self.stop_loss[0] > 0:
            self.balance_EUR = self.balance_USD / self.stop_loss[0]
            self.balance_USD = 0
            return self.balance_EUR, self.balance_USD, True
        else:
            return self.balance_EUR, self.balance_USD, False
