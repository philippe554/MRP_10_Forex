import time
import warnings

import numpy as np
import tensorflow as tf

from App.Library.lstm.Forex import Forex as ForexClass
from App.Library.lstm.PSO import PSO as PSO

l1Size = 40
l2Size = 30
lstmSize = 20
outputSize = 2
sequenceSize = 60
batchSize = 100
amountOfParticles = 120
amountOfEpochs = 100

forex = ForexClass(batchSize, sequenceSize, outputSize)
inputSize = len(forex.technical_indicators)

x = tf.placeholder("float", [None, sequenceSize, inputSize])

variables = {
    'l1': tf.Variable(tf.random_normal([inputSize, l1Size])),
    'l1b': tf.Variable(tf.random_normal([l1Size])),
    'l2': tf.Variable(tf.random_normal([l1Size, l2Size])),
    'l2b': tf.Variable(tf.random_normal([l2Size])),
    'l3': tf.Variable(tf.random_normal([lstmSize, outputSize])),
    'l3b': tf.Variable(tf.random_normal([outputSize]))
}

path_to_save = "C:/Users/Rodrigo/Google Drive/Group 10_ Forecast the FOREX market/Model_Weights/Rodrigo"
path_to_save += input("Name of the folder to load/save the weights: ") + "/model.ckpt"


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

# This is Phillipe's version of LSTM I do not know how to treat the windows

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for e in range(amountOfEpochs):
#         w = pso.get_particles()
#         X, price = forex.get_X()
#         f = np.zeros(amountOfParticles)
#
#         for p in range(amountOfParticles):
#             # set the parameters for this particle
#             ws = np.split(w[p,:], np.cumsum(variableSizes))[:-1]
#             for i in range(len(ws)):
#                 variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)
#
#             # small x is the placeholder of the tensorflow graph
#             # big X is the sample data of the Forex.py class
#             Y = sess.run(y, feed_dict={x: X})
#
#             f[p] = forex.calculate_profit(price, Y)
#
#         # negate profit, because PSO is cost based
#         # TODO: CHECK IF THIS IS THE P THAT THE METHOD USES
#         pso.update(-f, p)
#
#         print("Epoch", e, "finished with avg profit:", np.mean(f))

# Rodrigo created this modified version of LSTM, but needs to be checked

with tf.Session() as sess:
    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    try:
        saver.restore(sess, path_to_save)
    except:
        warnings.warn("New model created since it was not possible to load")
        sess.run(init_op)

    number_of_batches = round(forex.db_size / (sequenceSize * batchSize))
    print("The number of batches per epoch is", number_of_batches)

    for e in range(amountOfEpochs):
        forex.restart_offset_random()
        start_time = time.time()

        for batches in range(number_of_batches):
            w = pso.get_particles()
            X, price = forex.get_X()
            f = np.zeros(amountOfParticles)

            for p in range(amountOfParticles):
                # set the parameters for this particle
                ws = np.split(w[p, :], np.cumsum(variableSizes))[:-1]
                for i in range(len(ws)):
                    variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

                # small x is the placeholder of the tensorflow graph
                # big X is the sample data of the Forex.py class
                Y = sess.run(y, feed_dict={x: X})

                f[p] = forex.calculate_profit(price, Y)

            # negate profit, because PSO is cost based
            # TODO: CHECK IF THIS IS THE P THAT THE METHOD USES
            pso.update(-f, p)
            print("Iteration", batches, "finished with avg profit:", round(np.mean(f), 5))
            if batches % 100 == 0:
                print("Model saved")
                save_path = saver.save(sess, path_to_save)

        t_time = int(time.time() - start_time)
        minutes = int(t_time / 60)
        seconds = t_time % 60
        print("Epoch", e, "finished with avg profit:", round(np.mean(f), 5),
              "in", minutes, "minutes", seconds, "seconds")
