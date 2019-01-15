import random

from App.Library.lstm.ForexBase import *

scaler = 1000.0

trainPeriod = 60 * 24 * 700

class ForexGD(ForexBase):

    def get_X_train(self):
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            offset = int(random.random() * (trainPeriod - self.sequence_size - 15))

            getFrom = self.train_size - trainPeriod + offset
            getTo = self.train_size - trainPeriod + offset + self.sequence_size

            X[batch, :, :] = self.TA_train[getFrom : getTo, :]
            finalPrice = self.price_train[getTo, 1]
            price[batch, 0] = (self.price_train[getTo + 15, 1] - finalPrice) * scaler

        return X, price

    def get_X_test(self, batch_size = -1):
        if batch_size == -1:
            bs = self.batch_size
        else:
            bs = batch_size

        X = np.zeros((bs, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((bs, 1))

        for batch in range(bs):
            offset = int(random.random() * (self.test_size - self.sequence_size - 15))

            X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
            finalPrice = self.price_test[offset + self.sequence_size, 1]
            price[batch, 0] = (self.price_test[offset + self.sequence_size + 15, 1] - finalPrice) * scaler

        return X, price

    def reset_stats(self):
        return {}


