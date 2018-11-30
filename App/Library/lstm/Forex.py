import numpy as np


class Forex:
    def getX(self):
        X = np.random.rand(batchSize, sequenceSize, inputSize)
        price = np.random.rand(batchSize, sequenceSize)
        return X, price

    def calcProfit(self, price, Y):
        assert list(np.shape(price)) == [batchSize, sequenceSize]
        assert list(np.shape(Y)) == [batchSize, sequenceSize, outputSize]

        return np.random.rand(1)
