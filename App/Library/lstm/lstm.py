import numpy as np
import tensorflow as tf
from App.Library.lstm.FOREX import FOREX as FOREX
from App.Library.lstm.PSO import PSO as PSO

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
    'l3': tf.Variable(tf.random_normal([lstmSize, outputSize])),
    'l3b': tf.Variable(tf.random_normal([outputSize]))
}

def check(M, l):
    if M.get_shape().as_list() != l:
        print(M.get_shape().as_list())
        assert False

def batchMatMul(M, N):
    return tf.reshape(tf.reshape(M, [-1, M.get_shape()[-1]]) @ N, [-1, M.get_shape()[-2], N.get_shape()[-1]])

def buildNN(x):
    # x is input of neural network, size: (Batch size * Amount of timesteps * amount of technical indicators)

    # Feed forward layer. (batchMatMul is a TF trick to not have to split it)
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
        X, price = forex.getX()
        f = np.zeros(amountOfParticles)

        for p in range(amountOfParticles):
            # set the parameters for this particle
            ws = np.split(w[p,:], np.cumsum(variableSizes))[:-1]
            for i in range(len(ws)):
                variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

            # small x is the placeholder of the tensorflow graph
            # big X is the sample data of the FOREX.py class
            Y = sess.run(y, feed_dict={x: X})

            f[p] = forex.calcProfit(price, Y)

        # negate profit, because PSO is cost based
        pso.update(-f)

        print("Epoch", e, "finished with avg profit:", np.mean(f))

