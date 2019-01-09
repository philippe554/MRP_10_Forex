import pickle
import time

import numpy as np
import tensorflow as tf

from App.Library.Settings import settings
from App.Library.lstm.ForexRandom import ForexRandom
from App.Library.lstm.ForexSeq import ForexSeq
from App.Library.lstm.PSO import PSO

l1Size = 40
l2Size = 30
lstmSize = 20
outputSize = 2
sequenceSize = 180
batchSize = 50
amountOfParticles = 100
amountOfEpochs = 100


def forex_type():
    type = input("Type of Forex class to use, random or sequential? (1/2)")
    if type == 1:
        forex = ForexRandom(batchSize, sequenceSize, outputSize)
    else:
        forex = ForexSeq(batchSize, sequenceSize, outputSize)

    return forex


if settings.useParameters:
    if settings.forexType == "random":
        forex = ForexRandom(batchSize, sequenceSize, outputSize)
    elif settings.forexType == "seq":
        forex = ForexSeq(batchSize, sequenceSize, outputSize)
    else:
        forex = forex_type()
else:
    forex = forex_type()

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

    return tf.round(x)


y = buildNN(x)

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("Variables:", variableSizes, "Total:", np.sum(variableSizes))

if settings.useParameters:
    if settings.newModel:
        pso = PSO(amountOfParticles, np.sum(variableSizes))
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
        pso = PSO(amountOfParticles, np.sum(variableSizes))
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
                pso = PSO(amountOfParticles, np.sum(variableSizes))
                print("New PSO created")
            else:
                raise Exception(e)


def load_particle(sess, w, p):
    ws = np.split(w[p, :], np.cumsum(variableSizes))[:-1]
    for i in range(len(ws)):
        variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)


def run_model_test(sess, X, price):
    w = pso.get_best_particle()
    ws = np.split(w, np.cumsum(variableSizes))[:-1]

    for i in range(len(ws)):
        variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

    Y = sess.run(y, feed_dict={x: X})
    f = forex.calculate_profit_test(price, Y)

    return f


def run_model(sess, X, price):
    w = pso.get_particles()
    f = np.zeros(amountOfParticles)
    n_positions = np.zeros(amountOfParticles)

    for p in range(amountOfParticles):
        load_particle(sess, w, p)
        Y = sess.run(y, feed_dict={x: X})
        if test:
            f = forex.calculate_profit_test(price, Y)
        else:
            f[p], n_positions[p] = forex.calculate_profit(price, Y)

    return f, n_positions


def debug_output(meta, f, n_positions):
    print(meta, "avg profit:", np.mean(f), "avg pos:", np.mean(n_positions), pso.getStats())


def train_step(sess, e, b):
    X, price = forex.get_X_train()

    f, n_positions = run_model(sess, X, price)

    pso.update(-f)

    debug_output("Train " + str(e) + "-" + str(b) + ":", f, n_positions)


def test_step(sess, out_test):
    X, price = forex.get_X_test()

    f = run_model_test(sess, X, price)

    return f


def save_model():
    with open(path_to_save + '/model_parameters.pkl', 'wb') as output:
        pickle.dump(pso, output)
    print("Model saved in folder", path_to_save + '/model_parameters.pkl')


def train():
    with tf.Session() as sess:
        number_of_batches = round(forex.train_size / (sequenceSize * batchSize))
        print("The number of batches per epoch is", number_of_batches)

        for e in range(amountOfEpochs):
            forex.restart_offset_random()
            start_time = time.time()

            for b in range(number_of_batches):
                train_step(sess, e, b)

                if b % 50 == 0 and b > 0:
                    test_step(sess, False)
                    save_model()

            t_time = int(time.time() - start_time)
            minutes = int(t_time / 60)
            seconds = t_time % 60
            print("Epoch", e, "finished in", minutes, "minutes", seconds, "seconds")


def test():
    with tf.Session() as sess:
        number_of_batches = round(forex.test_size - sequenceSize)
        print("The number of batches is", number_of_batches)
        start_time = time.time()
        for e in range(number_of_batches):
            test_step(sess, True)

        t_time = int(time.time() - start_time)
        minutes = int(t_time / 60)
        seconds = t_time % 60
        print("Testing finished in", minutes, "minutes", seconds, "seconds")
        print("Money after testing:", forex.money, "number of positions:", forex.n_positions)


if settings.useParameters and settings.test:
    print("Testing the model...")
    test()
else:
    print("Training the model...")
    train()
