import pickle
import time

import numpy as np
import tensorflow as tf

from App.Library.Settings import settings

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

#path_to_save = "C:/Users/Rodrigo/checkpoints"
#folder = input("Name of the folder to load/save the weights: ")
#if folder:
#    path_to_save += "/" + folder


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

# newPSO = input("Do you want to load the previous PSO if exists? (y/n) ").lower() == "n"
#
# if newPSO:
#     pso = PSO(amountOfParticles, np.sum(variableSizes))
#     print("New PSO created")
# else:
#     try:
#         with open(path_to_save + '/model_parameters.pkl', 'rb') as model:
#             pso = pickle.load(model)
#         print("PSO loaded")
#     except Exception as e:
#         create_new = input(
#             "It was not possible to load the PSO, do you want to continue with a new PSO? (y/n) ").lower() == "y"
#         if create_new:
#             pso = PSO(amountOfParticles, np.sum(variableSizes))
#             print("New PSO created")
#         else:
#             raise Exception(e)

with tf.Session() as sess:
    # Add ops to save and restore all the variables.
    #saver = tf.train.Saver()

    #try:
    #    saver = tf.train.import_meta_graph(path_to_save + '/model.meta')
    #    saver.restore(sess, tf.train.latest_checkpoint(path_to_save))
    #    print("Model restored successfully")
    #except:
    #    warnings.warn("New model created since it was not possible to load")
    #    # Add an op to initialize the variables.
    #    sess.run(tf.global_variables_initializer())

    number_of_batches = round(forex.db_size / (sequenceSize * batchSize))
    print("The number of batches per epoch is", number_of_batches)

    for e in range(amountOfEpochs):
        #forex.restart_offset_random()
        start_time = time.time()
        avg = []

        for batches in range(number_of_batches):
            w = pso.get_particles()
            X, price = forex.get_X()
            f = np.zeros(amountOfParticles)
            n_positions = np.zeros(amountOfParticles)
            for p in range(amountOfParticles):
                # set the parameters for this particle
                ws = np.split(w[p, :], np.cumsum(variableSizes))[:-1]
                for i in range(len(ws)):
                    variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

                # small x is the placeholder of the tensorflow graph
                # big X is the sample data of the Forex.py class
                Y = sess.run(y, feed_dict={x: X})

                f[p], n_positions[p] = forex.calculate_profit(price, Y)

            # negate profit, because PSO is cost based
            pso.update(-f)
            new_avg = round(np.mean(f), 5)
            avg.append(new_avg)
            print("Iteration", batches,
                  "finished with avg profit: {:,} and avg of {:,} positions opened".format(new_avg,
                                                                                           round(np.mean(n_positions),
                                                                                                 2)))
            if batches % 50 == 0 and batches > 0:
                #save_path = saver.save(sess, path_to_save + "/model")
                with open(settings.modelPath + '/model_parameters.pkl', 'wb') as output:
                    pickle.dump(pso, output)
                print("Model saved in folder", settings.modelPath + '/model_parameters.pkl')

        t_time = int(time.time() - start_time)
        minutes = int(t_time / 60)
        seconds = t_time % 60
        print("Epoch", e, "finished with avg profit: {:,}".format(np.mean(avg)),
              "in", minutes, "minutes", seconds, "seconds")
