import random

from App.Library.lstm.ForexBase import *

scaler = 1000.0

class ForexGD(ForexBase):

    def get_X_train(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            offset = int(random.random() * (self.train_size - self.sequence_size - 10))

            X[batch, :, :] = self.TA_train[offset: (offset + self.sequence_size), :]
            finalPrice = self.price_train[offset + self.sequence_size, 1]
            price[batch, 0] = (self.price_train[offset + self.sequence_size + 10, 1] - finalPrice) * scaler

        return X, price

    def get_X_test(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            offset = int(random.random() * (self.test_size - self.sequence_size - 10))

            X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
            finalPrice = self.price_test[offset + self.sequence_size, 1]
            price[batch, 0] = (self.price_test[offset + self.sequence_size + 10, 1] - finalPrice) * scaler

        return X, price

    def reset_stats(self):
        return {}


