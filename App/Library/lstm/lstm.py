import numpy as np
import tensorflow as tf

inputSize = 50
l1Size = 40
l2Size = 30
lstmSize = 20
outputSize = 2
sequenceSize = 60
batchSize = 100
amountOfParticles = 120
amountOfEpochs = 10

x = tf.placeholder("float", [None, sequenceSize, inputSize])

variables = {
    'l1': tf.Variable(tf.random_normal([inputSize, l1Size])),
    'l1b': tf.Variable(tf.random_normal([l1Size])),
    'l2': tf.Variable(tf.random_normal([l1Size, l2Size])),
    'l2b': tf.Variable(tf.random_normal([l2Size])),
    'l3': tf.Variable(tf.random_normal([20, 2])),
    'l3b': tf.Variable(tf.random_normal([2]))
}

def check(M, l):
    if M.get_shape().as_list() != l:
        print(M.get_shape().as_list())
        assert False

def batchMatMul(M, N):
    return tf.reshape(tf.reshape(M, [-1, M.get_shape()[-1]]) @ N, [-1, M.get_shape()[-2], N.get_shape()[-1]])

def buildNN(x):
    x = tf.nn.relu(batchMatMul(x, variables['l1']) + variables['l1b'])
    check(x, [None, sequenceSize, l1Size])

    x = tf.nn.relu(batchMatMul(x, variables['l2']) + variables['l2b'])
    check(x, [None, sequenceSize, l2Size])

    x = tf.unstack(x, sequenceSize, 1)
    cell1 = tf.nn.rnn_cell.LSTMCell(lstmSize)
    outputs, states = tf.nn.static_rnn(cell1, x, dtype=tf.float32)
    x = tf.stack(outputs, 1)
    check(x, [None, sequenceSize, lstmSize])

    x = tf.nn.sigmoid(batchMatMul(x, variables['l3']) + variables['l3b'])
    check(x, [None, sequenceSize, outputSize])

    return x

class PSO:
    def __init__(self, amountOfParticles, dims):
        self.amountOfParticles = amountOfParticles
        self.dims = dims

        # TODO: properly select random distributions and initial cost
        self.pos = np.random.rand(self.amountOfParticles, self.dims)
        self.vel = np.random.rand(self.amountOfParticles, self.dims)

        self.bestPos = self.pos
        self.bestCost = np.full(self.amountOfParticles, 1000)

        self.bestSwarmPos = self.pos[0] # TODO: should not be the first
        self.bestSwarmCost = 1000

        # Just some random hyper parameters i found somewhere
        self.omega = 0.7
        self.phiP = 2
        self.phiG = 2

    def getParticles(self):
        assert list(np.shape(self.pos)) == [self.amountOfParticles, self.dims]
        return self.pos

    def update(self, cost):
        assert len(cost) == self.amountOfParticles

        rp = np.random.rand(self.amountOfParticles, self.dims)
        rg = np.random.rand(self.amountOfParticles, self.dims)
        self.vel = self.omega * self.vel + self.phiP * rp * (self.bestPos - self.pos) + self.phiG * rg * (self.bestSwarmPos - self.pos)
        self.pos = self.pos + self.vel

        for i in range(self.amountOfParticles):
            if cost[i] < self.bestCost[i]:
                self.bestPos[p,:] = self.pos[p,:]
                self.bestCost[i] = cost[i]

                if cost[i] < self.bestSwarmCost:
                    self.bestSwarmPos = self.pos[p,:]
                    self.bestSwarmCost = cost[i]

class FOREX:
    def getX(self):
        return np.random.rand(batchSize, sequenceSize, inputSize)

    def calcProfit(self, x, y):
        assert list(np.shape(x)) == [batchSize, sequenceSize, inputSize]
        assert list(np.shape(y)) == [batchSize, sequenceSize, outputSize]
        return np.random.rand(1)

y = buildNN(x)

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("Variables:", variableSizes, "Total:", np.sum(variableSizes))

pso = PSO(amountOfParticles, np.sum(variableSizes))

forex = FOREX()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(amountOfEpochs):
        w = pso.getParticles()
        X = forex.getX()
        f = np.zeros(amountOfParticles)

        for p in range(amountOfParticles):
            # set the parameters for this particle
            ws = np.split(w[p,:], np.cumsum(variableSizes))[:-1]
            for i in range(len(ws)):
                variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

            Y = sess.run(y, feed_dict={x: X})

            f[p] = forex.calcProfit(X, Y)

        # negate profit, because PSO is cost based
        pso.update(-f)

        print("Epoch", e, "finished with avg profit:", np.mean(f))

