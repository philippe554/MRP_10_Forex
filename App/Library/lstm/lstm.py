# Fix pythonpath if executing on cluster
import sys

if any("rwthfs" in s for s in sys.path):
    sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/.local/lib/python3.6/site-packages')
    sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/MRP_10_Forex/')

import datetime
import pickle
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from App.Library.Settings import settings
from App.Library.lstm.ForexOverlap import ForexOverlap
from App.Library.lstm.ForexRandom import ForexRandom
from App.Library.lstm.ForexSeq import ForexSeq
from App.Library.lstm.PSO import PSO

if settings.useParameters:
    if settings.newModel:
        pso = PSO(settings.forexType)
        print("New PSO created")
    else:
        try:
            with open(settings.modelPath + '/model_parameters.pkl', 'rb') as model:
                pso = pickle.load(model)
            print("PSO loaded")
        except Exception as e:
            print("Failed to load PSO")
            exit()

    path_to_save = settings.modelPath
else:
    path_to_save = "C:/Users/Rodrigo/checkpoints"
    folder = input("Name of the folder to load/save the weights: ")
    if folder:
        path_to_save += "/" + folder

    newPSO = input("Do you want to load the previous PSO if exists? (y/n) ").lower() == "n"

    if newPSO:
        pso = PSO()
        print("New PSO created")
    else:
        try:
            with open(path_to_save + '/model_parameters.pkl', 'rb') as model:
                pso = pickle.load(model)
            print("PSO loaded")
        except Exception as e:
            create_new = input(
                "It was not possible to load the PSO, do you want to continue with a new PSO? (y/n) ").lower() == "y"
            if create_new:
                pso = PSO()
                print("New PSO created")
            else:
                raise Exception(e)

pso.print_hyper_parameters()


def forex_type():
    type = input("Type of Forex class to use, random, sequential or overlap? (1/2/3)")
    if type == 1:
        forex = ForexRandom(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
    elif type == 2:
        forex = ForexSeq(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
    else:
        forex = ForexOverlap(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)

    return forex


if settings.useParameters:
    if settings.forexType == "random":
        forex = ForexRandom(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
    elif settings.forexType == "overlap":
        forex = ForexOverlap(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
    elif settings.forexType == "seq":
        forex = ForexSeq(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
    else:
        forex = forex_type()
else:
    forex = forex_type()
forex.reset_stats()

inputSize = len(forex.technical_indicators)

variables = {
    'l1': tf.Variable(tf.random_normal([inputSize, pso.l1Size])),
    'l1b': tf.Variable(tf.random_normal([pso.l1Size])),
    'l2': tf.Variable(tf.random_normal([pso.l1Size, pso.l2Size])),
    'l2b': tf.Variable(tf.random_normal([pso.l2Size])),
    'l3': tf.Variable(tf.random_normal([pso.lstmSize, pso.outputSize])),
    'l3b': tf.Variable(tf.random_normal([pso.outputSize]))
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
    check(x, [None, pso.sequenceSize, pso.l1Size])

    x = tf.nn.relu(batchMatMul(x, variables['l2']) + variables['l2b'])
    check(x, [None, pso.sequenceSize, pso.l2Size])

    x = tf.unstack(x, pso.sequenceSize, 1)
    cell1 = tf.nn.rnn_cell.LSTMCell(pso.lstmSize)
    outputs, states = tf.nn.static_rnn(cell1, x, dtype=tf.float32)
    x = tf.stack(outputs, 1)
    check(x, [None, pso.sequenceSize, pso.lstmSize])

    x = tf.nn.sigmoid(batchMatMul(x, variables['l3']) + variables['l3b'])
    check(x, [None, pso.sequenceSize, pso.outputSize])

    return tf.round(x)


def buildNNOverlap(x):
    check(x, [None, pso.sequenceSize + pso.sequenceOverlap, inputSize])

    x = tf.stack([x[:, i:i + pso.sequenceSize, :] for i in range(pso.sequenceOverlap)], axis=1)
    check(x, [None, pso.sequenceOverlap, pso.sequenceSize, inputSize])

    # Merge the batch dimension with the overlap dimension, for tensorflow they are both batches
    x = tf.reshape(x, shape=[-1, pso.sequenceSize, inputSize])
    check(x, [None, pso.sequenceSize, inputSize])

    x = buildNN(x)
    check(x, [None, pso.sequenceSize, pso.outputSize])

    # Unfold the merge
    x = tf.reshape(x, shape=[-1, pso.sequenceOverlap, pso.sequenceSize, pso.outputSize])
    check(x, [None, pso.sequenceOverlap, pso.sequenceSize, pso.outputSize])

    x = x[:, :, -1, :]
    check(x, [None, pso.sequenceOverlap, pso.outputSize])

    return x


if settings.forexType == "overlap":
    x = tf.placeholder("float", [None, pso.sequenceSize + pso.sequenceOverlap, inputSize])
    y = buildNNOverlap(x)
else:
    x = tf.placeholder("float", [None, pso.sequenceSize, inputSize])
    y = buildNN(x)

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("Variables:", variableSizes, "Total:", np.sum(variableSizes))

if settings.newModel:
    pso.reset_particles(np.sum(variableSizes))


def load_particle(sess, w, p):
    ws = np.split(w[p, :], np.cumsum(variableSizes))[:-1]
    for i in range(len(ws)):
        variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)


def run_model_test(sess, X, price, draw):
    w = pso.get_best_particle()
    ws = np.split(w, np.cumsum(variableSizes))[:-1]

    for i in range(len(ws)):
        variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

    Y = sess.run(y, feed_dict={x: X})
    f = forex.calculate_profit_test(price, Y, draw)

    return f


def run_model(sess, X, price):
    w = pso.get_particles()
    f = np.zeros(pso.amountOfParticles)
    n_positions = np.zeros(pso.amountOfParticles)

    for p in range(pso.amountOfParticles):
        load_particle(sess, w, p)
        Y = sess.run(y, feed_dict={x: X})
        f[p], n_positions[p] = forex.calculate_profit(price, Y)

    return f, n_positions


def debug_output(meta, f, n_positions, stats=None):
    if settings.forexType == 'overlap':
        print(meta, "avg cost:", "%.7f" % -np.mean(f), ", avg trades:", "%.3f" % np.mean(n_positions), pso.getStats(),
              stats)
    else:
        print(meta, "avg profit:", "%.7f" % np.mean(f), "avg trades:", "%.3f" % np.mean(n_positions), pso.getStats())


def train_step(sess, e, b):
    d1 = datetime.datetime.now()

    X, price = forex.get_X_train()

    f, n_positions = run_model(sess, X, price)
    stats = forex.reset_stats()

    pso.update(-f)

    d2 = datetime.datetime.now()
    delta = d2 - d1
    debug_output("Train " + str(e) + "-" + str(b) + " [" + "%.2f" % (delta.total_seconds() * 1000) + "ms] :", f,
                 n_positions, stats)


def test_step(sess, draw=False):
    # Run test on a larger batch
    test_size = 3000
    X, price = forex.get_X_test(test_size)

    if settings.forexType == 'seq':
        # TODO show seq test output
        f = run_model_test(sess, X, price, draw)
        print("=== TEST === TODO: show seq test output")
        return f
    else:
        f, n_positions = run_model_test(sess, X, price, draw)
        stats = forex.reset_stats()

        avg_profit = np.mean(f)
        avg_trades = np.mean(n_positions)
        profit_per_trade = avg_profit / avg_trades
        print('\033[32m'+"=== TEST === best particle on", test_size, "batches:", "avg profit per trade:", "%.5f" % profit_per_trade, "avg profit:", "%.5f" % avg_profit, " avg trades:", "%.3f" % avg_trades, stats, '\033[0m')
        return f, n_positions


def save_model():
    with open(path_to_save + '/model_parameters.pkl', 'wb') as output:
        pickle.dump(pso, output)
    print("Model saved in folder", path_to_save + '/model_parameters.pkl')


test_every = 5


def train():
    with tf.Session() as sess:
        number_of_batches = round(forex.train_size / (pso.sequenceSize * pso.batchSize))
        print("The number of batches per epoch is", number_of_batches)

        for e in range(pso.amountOfEpochs):
            forex.restart_offset_random()
            start_time = time.time()

            for b in range(number_of_batches):
                train_step(sess, e, b)

                if b % test_every == 0 and b > 0:
                    test_step(sess, draw=True)
                    save_model()
            t_time = int(time.time() - start_time)
            minutes = int(t_time / 60)
            seconds = t_time % 60
            print("Epoch", e, "finished in", minutes, "minutes", seconds, "seconds")


def test():
    with tf.Session() as sess:

        start_time = time.time()
        total_profit = 0
        prices = []
        finished = False
        while not finished:
            # TODO: send capital as input to calculate profit
            X, price = forex.get_X_test()
            f, n_positions = run_model_test(sess, X, price, True)
            debug_output("Test:", f, n_positions)

            for p in range(len(price)):
                prices.extend(price[p])
            if forex.test_offset + forex.batch_size * (forex.sequence_size + forex.sequence_overlap) > forex.test_size:
                finished = True

        t_time = int(time.time() - start_time)
        minutes = int(t_time / 60)
        seconds = t_time % 60
        print("Total profit after testing:", total_profit)
        print("Testing finished in", minutes, "minutes", seconds, "seconds")

        # plt.subplot(2, 1, 1)
        # plt.plot(forex.price_test)
        # plt.title("Data from price_test")
		#
        # plt.subplot(2, 1, 2)
        # plt.title("Data from get_X_test")
        # plt.plot(prices)
        # plt.show()


if settings.useParameters and settings.test:
    if settings.newModel:
        raise Exception("You cannot train a new model")
    print("Testing the model...")

    test()
else:
    print("Training the model...")
    train()
