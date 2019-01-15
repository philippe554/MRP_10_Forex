import datetime
import pickle
import time
import traceback

import numpy as np
import tensorflow as tf

from App.Library.lstm.ForexGD import ForexGD

# EXAMPLE : -p -i "C:\dev\data\forex\data\price_hist_ta.db" -c "C:\dev\data\forex\model"

l1Size = 30
l2Size = 20
lstmSize = 20
outputSize = 1
sequenceSize = 60
batchSize = 500
epochs = 200

forex = ForexGD(batchSize, sequenceSize, -1, outputSize)
inputSize = len(forex.technical_indicators)

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

	x = tf.nn.tanh(batchMatMul(x, variables['l3']) + variables['l3b'])
	check(x, [None, sequenceSize, outputSize])

	x = x[:,-1,:]
	check(x, [None, outputSize])

	return x

x = tf.placeholder("float", [None, sequenceSize, inputSize])
y = tf.placeholder("float", [None, outputSize])

Y = buildNN(x)

lossCalc = tf.reduce_mean(tf.square(Y - y))

trainStep = tf.train.AdamOptimizer(1e-4).minimize(lossCalc)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	batchesPerEpoch = int(forex.trainPeriod / (batchSize * inputSize))
	print("Batches per epoch:", batchesPerEpoch)

	for e in range(epochs):
		d1 = datetime.datetime.now()

		lossList = []
		priceDiffList = []
		hitList = []
		hitListPred = []
		testAcc = []
		for i in range(batchesPerEpoch):
			X, price = forex.get_X_train()
			priceDiffList.append(np.abs(price))
			hitList.append(np.mean(np.abs(price) > 1))

			loss, _, pred = sess.run([lossCalc, trainStep, Y], feed_dict={x: X, y: price})

			hitListPred.append(np.mean(np.abs(pred) > 0.4))

			testAcc.extend(price[pred > 0.4] > 0.2)
			testAcc.extend(price[pred < -0.4] < -0.2)

			lossList.append(loss)

		X, price = forex.get_X_test(2000)
		loss, pred = sess.run([lossCalc, Y], feed_dict={x: X, y: price})

		trainHitFreq = np.mean(pred > 0.4)

		trainAccProfit = []
		trainAccProfit.extend(price[pred > 0.4] > 0.2)
		trainAccProfit.extend(price[pred < -0.4] < -0.2)

		trainAccNoLoss = []
		trainAccNoLoss.extend(price[pred > 0.4] > 0.04)
		trainAccNoLoss.extend(price[pred < -0.4] < -0.04)

		d2 = datetime.datetime.now()
		time = "%.2f" % ((d2 - d1).total_seconds() * 1000) + "ms"

		print(e, time, ": Loss:", np.mean(lossList), "-", loss, "Acc profit:", np.mean(testAcc), "-", np.mean(trainAccProfit), "No loss:", np.mean(trainAccNoLoss), "Freq:", trainHitFreq)