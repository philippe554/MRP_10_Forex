import random

from App.Library.lstm.ForexBase import *


class ForexRandom(ForexBase):

    def get_X_train(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, self.sequence_size))

        for batch in range(self.batch_size):
            offset = int(random.random() * (self.train_size - self.sequence_size))

            X[batch, :, :] = self.TA_train[offset: (offset + self.sequence_size), :]
            price[batch, :] = self.price_train[offset: (offset + self.sequence_size), 1]

        return X, price

    def get_X_test(self, batch_size):
        X = np.zeros((batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((batch_size, self.sequence_size))

        for batch in range(batch_size):
            offset = int(random.random() * (self.test_size - self.sequence_size))

            X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
            price[batch, :] = self.price_test[offset: (offset + self.sequence_size), 1]

        return X, price

    def calculate_profit(self, price, Y):
        batch_size = len(price)
        position_cost = np.ones(batch_size) * 0.0002
        positions_total = 0.0
        pips_gained = np.zeros(batch_size)
        position_long = np.zeros(batch_size)
        position_short = np.zeros(batch_size)
        position_long_open = np.zeros(batch_size)
        position_short_open = np.zeros(batch_size)

        for i in range(self.sequence_size):
            open_long_index = Y[:, i, 0] > position_long
            position_long_open[open_long_index] = price[open_long_index, i]
            position_long[open_long_index] = 1

            close_long_index = Y[:, i, 0] < position_long
            pips_gained[close_long_index] += position_long_open[close_long_index] - price[close_long_index, i] - \
                                             position_cost[close_long_index]
            position_long[close_long_index] = 0
            positions_total += np.sum(close_long_index)

            open_short_index = Y[:, i, 1] > position_short
            position_short_open[open_short_index] = price[open_short_index, i]
            position_short[open_short_index] = 1

            close_short_index = Y[:, i, 1] < position_short
            pips_gained[close_short_index] += price[close_short_index, i] - position_short_open[close_short_index] - \
                                              position_cost[close_short_index]
            position_short[close_short_index] = 0
            positions_total += np.sum(close_short_index)

        close_long_index = 0 < position_long
        pips_gained[close_long_index] += position_long_open[close_long_index] - price[close_long_index, i] - \
                                         position_cost[close_long_index]
        position_long[close_long_index] = 0
        positions_total += np.sum(close_long_index)

        close_short_index = 0 < position_short
        pips_gained[close_short_index] += price[close_short_index, i] - position_short_open[close_short_index] - \
                                          position_cost[close_short_index]
        position_short[close_short_index] = 0
        positions_total += np.sum(close_short_index)

        return np.mean(pips_gained), positions_total / batch_size

    def calculate_profit_check(self, price, Y):
        position_cost = 0.0002
        positions_total = 0.0
        pips_gained = np.zeros(self.batch_size)

        for j in range(self.batch_size):
            position_long = 0
            position_short = 0
            position_long_open = 0
            position_short_open = 0
            for i in range(self.sequence_size):
                if Y[j, i, 0] > position_long:
                    position_long_open = price[j, i]
                    position_long = 1
                if Y[j, i, 0] < position_long:
                    pips_gained[j] += position_long_open - price[j, i] - position_cost
                    position_long = 0
                    positions_total += 1

                if Y[j, i, 1] > position_short:
                    position_short_open = price[j, i]
                    position_short = 1
                if Y[j, i, 1] < position_short:
                    pips_gained[j] += price[j, i] - position_short_open - position_cost
                    position_short = 0
                    positions_total += 1

            if Y[j, i, 0] < position_long:
                pips_gained[j] += position_long_open - price[j, i] - position_cost
                positions_total += 1

            if Y[j, i, 1] < position_short:
                pips_gained[j] += price[j, i] - position_short_open - position_cost
                positions_total += 1

        return np.mean(pips_gained), positions_total / self.batch_size

    def calculate_profit_test(self, price, Y, draw):
        # TODO: Implement method
        return self.calculate_profit(price, Y)
