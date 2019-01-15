import random

from App.Library.lstm.ForexBase import *


class ForexRandom(ForexBase):

    def get_X_train(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, 4))

        for batch in range(self.batch_size):
            offset = int(random.random() * (self.train_size - self.sequence_size - 21))

            X[batch, :, :] = self.TA_train[offset: (offset + self.sequence_size), :]
            price[batch, 0] = self.price_train[offset + self.sequence_size + 2, 1]
            price[batch, 1] = self.price_train[offset + self.sequence_size + 5, 1]
            price[batch, 2] = self.price_train[offset + self.sequence_size + 10, 1]
            price[batch, 3] = self.price_train[offset + self.sequence_size + 20, 1]

        return X, price

    def get_X_test(self, batch_size):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, 4))

        for batch in range(self.batch_size):
            offset = int(random.random() * (self.test_size - self.sequence_size - 21))

            X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
            price[batch, 0] = self.price_test[offset + self.sequence_size + 2, 1]
            price[batch, 1] = self.price_test[offset + self.sequence_size + 5, 1]
            price[batch, 2] = self.price_test[offset + self.sequence_size + 10, 1]
            price[batch, 3] = self.price_test[offset + self.sequence_size + 20, 1]

        return X, price

    def reset_stats(self):
        return {}


