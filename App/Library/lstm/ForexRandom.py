import random
import numpy as np

from App.Library.lstm.ForexBase import *

class ForexRandom(ForexBase):
    def get_random_offset(self):
        return int(random.random() * (self.train_size - self.sequence_size))

    def get_X_train(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, self.sequence_size))

        for batch in range(self.batch_size):
            offset = self.get_random_offset()

            X[batch, :, :] = self.TA_train[offset : (offset + self.sequence_size), :]
            price[batch, :] = self.price_train[offset : (offset + self.sequence_size), 0]

        return X, price

    def get_X_test(self):
        assert False
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, self.sequence_size))

        for batch in range(self.batch_size):
            offset = self.get_random_offset()

            X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
            price[batch, :] = self.price_test[offset: (offset + self.sequence_size), 0]

        return X, price

    def calculate_profit(self, price, Y):
        position_cost = np.ones(self.batch_size) * 0.001
        positions_total = 0;
        pipsGained = np.zeros(self.batch_size)
        position_long = np.zeros(self.batch_size)
        position_short = np.zeros(self.batch_size)
        position_long_open = np.zeros(self.batch_size)
        position_short_open = np.zeros(self.batch_size)

        for i in range(self.sequence_size):
            open_long_index = Y[:, i, 0] > position_long
            position_long_open[open_long_index] = price[open_long_index, i]
            position_long[open_long_index] = 1

            close_long_index = Y[:, i, 0] < position_long
            pipsGained[close_long_index] += position_long_open[close_long_index] - price[close_long_index, i] + \
                                            position_cost[close_long_index]
            position_long[close_long_index] = 0
            positions_total += np.sum(close_long_index);

            open_short_index = Y[:, i, 1] > position_short
            position_short_open[open_short_index] = price[open_short_index, i]
            position_short[open_short_index] = 1

            close_short_index = Y[:, i, 1] < position_short
            pipsGained[close_short_index] += price[close_short_index, i] - position_short_open[close_short_index] + \
                                             position_cost[close_short_index]
            position_short[close_short_index] = 0
            positions_total += np.sum(close_short_index);

        close_long_index = 0 < position_long
        pipsGained[close_long_index] += position_long_open[close_long_index] - price[close_long_index, i] + \
                                        position_cost[close_long_index]
        position_long[close_long_index] = 0
        positions_total += np.sum(close_long_index);

        close_short_index = 0 < position_short
        pipsGained[close_short_index] += price[close_short_index, i] - position_short_open[close_short_index] + \
                                         position_cost[close_short_index]
        position_short[close_short_index] = 0
        positions_total += np.sum(close_short_index);

        return np.mean(pipsGained), positions_total / self.batch_size