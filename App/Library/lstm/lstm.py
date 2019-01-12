# Fix pythonpath if executing on cluster
import sys

if any("rwthfs" in s for s in sys.path):
	sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/.local/lib/python3.6/site-packages')
	sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/MRP_10_Forex/')

drawEnabled = False
try:
	import matplotlib.pyplot as plt
	drawEnabled = True
except ImportError:
	print("Drawing plots is disabled, make sure you have the matplotlib module installed")

import datetime
import pickle
import time
import traceback

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from App.Library.Settings import settings
from App.Library.lstm.ForexOverlap import ForexOverlap
from App.Library.lstm.ForexRandom import ForexRandom
from App.Library.lstm.ForexSeq import ForexSeq
from App.Library.lstm.PSO import PSO

print(" === Loading PSO")
if settings.useParameters:
	if settings.newModel:
		pso = PSO(settings.forexType)
		print("     New PSO created")
	else:
		try:
			with open(settings.modelPath + '/model_parameters.pkl', 'rb') as model:
				pso = pickle.load(model)
			print("     PSO loaded from file")
		except Exception as e:
			traceback.print_exc()
			print("     Failed to load PSO, Exiting..")
			quit()

	path_to_save = settings.modelPath
else:
	path_to_save = "C:/Users/Rodrigo/checkpoints"
	folder = input("Name of the folder to load/save the weights: ")
	if folder:
		path_to_save += "/" + folder

	newPSO = input("Do you want to load the previous PSO if exists? (y/n) ").lower() == "n"

	if newPSO:
		pso = PSO(settings.forexType)
		print("     New PSO created")
	else:
		try:
			with open(path_to_save + '/model_parameters.pkl', 'rb') as model:
				pso = pickle.load(model)
			print("     PSO loaded")
		except Exception as e:
			create_new = input(
				"It was not possible to load the PSO, do you want to continue with a new PSO? (y/n) ").lower() == "y"
			if create_new:
				pso = PSO(settings.forexType)
				print("     New PSO created")
			else:
				raise Exception(e)

pso.print_hyper_parameters()

print(" === Initializing forexType")
def forex_type():
	type = input("Type of Forex class to use, random, sequential or overlap? (1/2/3)")
	if type == 1:
		print("     forexType: Random")
		forex = ForexRandom(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
	elif type == 2:
		print("     forexType: Sequential")
		forex = ForexSeq(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
	else:
		print("     forexType: Overlap")
		forex = ForexOverlap(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)

	return forex


if settings.useParameters:
	if settings.forexType == "random":
		print("     forexType: Random")
		forex = ForexRandom(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
	elif settings.forexType == "overlap":
		print("     forexType: Overlap")
		forex = ForexOverlap(pso.batchSize, pso.sequenceSize, pso.sequenceOverlap, pso.outputSize)
	elif settings.forexType == "seq":
		print("     forexType: Sequential")
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


print(" === Building model")
if settings.forexType == "overlap" and not settings.test:
	x = tf.placeholder("float", [None, pso.sequenceSize + pso.sequenceOverlap, inputSize])
	y = buildNNOverlap(x)
	print("     Overlap NN loaded")
else:
	x = tf.placeholder("float", [None, pso.sequenceSize, inputSize])
	y = buildNN(x)
	print("     NN loaded (no overlap)")

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("Variables:", variableSizes, "Total:", np.sum(variableSizes))

if settings.newModel:
	print("     Reset PSO particles")
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
	test_size = 3000  # Run test on a larger batch
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
		print('\033[32m' + "=== TEST === best particle on", test_size, "batches:", "avg profit per trade:",
			  "%.5f" % profit_per_trade, "avg profit:", "%.5f" % avg_profit, " avg trades:", "%.3f" % avg_trades, stats,
			  '\033[0m')
		return f, n_positions


def save_model():
	with open(path_to_save + '/model_parameters.pkl', 'wb') as output:
		pickle.dump(pso, output)
	print("Model saved in folder", path_to_save + '/model_parameters.pkl')

def simulate_real_test(sess, test_window):
	"""
		Simulates realtime trading using the provided data window
		current PSO best particle is used
		test_step uses the same window that is used during training, this should be a larger window and give a better estimate of real profit
	"""

	d1 = datetime.datetime.now()
	print('\033[94m' + "=== Real test: simulating real trading ===")

	# Retrieve the test data
	test_TA = forex.TA_test
	test_price = forex.price_test

	# Fit the window
	test_TA = test_TA[forex.test_size-test_window:]
	test_price = test_price[forex.test_size-test_window:]
	print("Test window: ", test_price[0,0].strftime("%Y-%m-%d %H:%I:%S"), " - ", test_price[-1,0].strftime("%Y-%m-%d %H:%I:%S"))

	# Load the best particle
	w = pso.get_best_particle()
	ws = np.split(w, np.cumsum(variableSizes))[:-1]
	for i in range(len(ws)):
		variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

	# Constants
	commission = 4  # Dollar per 100k traded
	capital = 50000
	transaction_fee = (capital / 100000) * commission

	# Parameters
	min_buy_signals = 1  # Wait for N buy signals before buying

	# Loop all test data
	position = 0
	bought = []
	sold = []
	profit = []
	total_profit = 0
	offset = 0
	num_buy = 0
	total_batches = test_window - pso.sequenceSize
	print("Starting test run on " + str(total_batches) + " consecutive batches")
	while offset < total_batches:
		if offset % 10000 == 0 and offset > 0:
			print("Progress: ", offset, "/", total_batches, " current profit: ", total_profit)

		# Get the next sequence
		X = [test_TA[offset:offset + pso.sequenceSize]]

		# Run the lstm on the sequence
		Y = sess.run(y, feed_dict={x: X})

		# Handle output (this can be different for each type, as long as the output is to buy or sell)
		buy, sell = forex.evaluate_output(Y)

		# Handle signals
		if position == 0 and buy and not sell:
			num_buy += 1
			if num_buy >= min_buy_signals:
				# Open a new position at the current rate (close)
				bought.append(offset + pso.sequenceSize)
				position = test_price[offset + pso.sequenceSize, 4]
		elif position != 0 and sell:
			# Close the position using the current rate (close)
			total_profit += (capital * (test_price[offset + pso.sequenceSize, 4] - position)) - transaction_fee
			sold.append(offset + pso.sequenceSize)
			position = 0
			num_buy = 0

		offset += 1
		profit.append(total_profit)

	# Debug plot
	if drawEnabled:
		plt.plot(test_price[pso.sequenceSize:, 4], 'k-')
		plt.plot(bought, test_price[bought, 4].tolist(), 'r+', label='buy signal')
		plt.plot(sold, test_price[sold, 4].tolist(), 'bx', label='buy signal')
		plt.legend()
		plt.ylabel("EUR/USD")
		plt.xlabel("2018")
		plt.title("Gross profit/loss: " + "%.3f" % total_profit)
		plt.show()

		plt.plot(profit, 'r-')
		plt.ylabel("Profit")
		plt.xlabel("Time")
		plt.title("Gross profit/loss: " + "%.3f" % total_profit)
		plt.show()

	d2 = datetime.datetime.now()
	delta = d2 - d1
	print("\n\tTest finished [" + "%.2f" % (delta.total_seconds() * 1000) + "ms]:" +
				 "\n\ttotal profit/loss: " + str(total_profit) +
				 "\n\ttotal trades: " + str(len(sold)) +
				 "\n\tavg profit per trade: " + str(total_profit/len(sold)) +
				 "\n\ttotal trade volume: " + str(capital*len(sold)) + '\n\n\033[0m')


def train():
	test_every = 5  # Run the test every N iterations
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
	"""
	The test run simulates real trading on the test dataset
	The test data is looped from beginning to end in 1 minute intervals
	This is how the loaded particle would perform as if it would run in realtime
	"""
	with tf.Session() as sess:
		# simulate_real_test(sess, forex.test_size)
		simulate_real_test(sess, 50000)

if settings.useParameters and settings.test:
	if settings.newModel:
		raise Exception("You cannot train a new model")
	print("Testing the model...")

	test()
else:
	print("Training the model...")
	train()
