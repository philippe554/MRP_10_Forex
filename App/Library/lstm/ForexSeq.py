import random
import numpy as np

from App.Library.lstm.ForexBase import *

class ForexSeq(ForexBase):
    def get_random_offset(self):
        return int(random.random() * (self.train_size - self.sequence_size))

    def get_X_train(self):
        X = np.zeros(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.zeros(self.batch_size, self.sequence_size)

        return X, price

    def get_X_test(self):
        assert False
        X = np.zeros(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.zeros(self.batch_size, self.sequence_size)

        return X, price

    def calculate_profit(self, price, Y):
        return 1

    def restart_offset_random(self):
        pass