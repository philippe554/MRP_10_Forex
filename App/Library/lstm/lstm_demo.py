print("Startup")

# Fix pythonpath if executing on cluster
import sys
if any("rwthfs" in s for s in sys.path):
	print("Expanding pythonpath")
	sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/.local/lib/python3.6/site-packages')
	sys.path.insert(0, '/rwthfs/rz/cluster/home/dh060408/MRP_10_Forex/')

import datetime as dt
import pickle
import time
import traceback

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from App.Library.Settings import settings
from App.Library.lstm.ForexDemo import ForexDemo
from App.Library.lstm.PSO import PSO
from App.Helpers.LiveTA import LiveTA

l1Size = 4
l2Size = 8
lstmSize = 6
outputSize = 2
sequenceSize = 30
sequenceOverlap = 30
batchSize = 1
amountOfParticles = 100
amountOfEpochs = 100

pso_path = "/Saved/10-1-2351"

print(" === Initializing forex class")
forex = ForexDemo(sequenceSize)
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

    x = tf.nn.sigmoid(batchMatMul(x, variables['l3']) + variables['l3b'])
    check(x, [None, sequenceSize, outputSize])

    return tf.round(x)

def buildNNOverlap(x):
    check(x, [None, sequenceSize + sequenceOverlap, inputSize])

    x = tf.stack([x[:,i:i+sequenceSize,:] for i in range(sequenceOverlap)], axis=1)
    check(x, [None, sequenceOverlap, sequenceSize, inputSize])

    # Merge the batch dimension with the overlap dimension, for tensorflow they are both batches
    x = tf.reshape(x, shape=[-1, sequenceSize, inputSize])
    check(x, [None, sequenceSize, inputSize])

    x = buildNN(x)
    check(x, [None, sequenceSize, outputSize])

    # Unfold the merge
    x = tf.reshape(x, shape=[-1, sequenceOverlap, sequenceSize, outputSize])
    check(x, [None, sequenceOverlap, sequenceSize, outputSize])

    x = x[:,:,-1,:]
    check(x, [None, sequenceOverlap, outputSize])

    return x

print(" === Building model")
if sequenceOverlap > 0:
	x = tf.placeholder("float", [None, sequenceSize + sequenceOverlap, inputSize])
	y = buildNNOverlap(x)
	print("     Overlap NN loaded")
else:
	x = tf.placeholder("float", [None, sequenceSize, inputSize])
	y = buildNN(x)
	print("     NN loaded")


print(" === Initializing model variables")
variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("      Variables:", variableSizes, "Total:", np.sum(variableSizes))

print(" === Loading PSO parameters")
try:
	with open(settings.modelPath + pso_path + '/model_parameters.pkl', 'rb') as model:
		pso = pickle.load(model)
	print("     PSO loaded")
except Exception as e:
	print("     Failed to load PSO, Exiting..")
	quit()

def debug_output(message):
	print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": ", message)

def terminate_all():
	quit()

with tf.Session() as sess:
	print(" === Loading best PSO particle")
	particle = pso.get_best_particle()
	ws = np.split(particle, np.cumsum(variableSizes))[:-1]
	for i in range(len(ws)):
		variables[i].load(ws[i].reshape(variables[i].get_shape().as_list()), sess)

	print(" === Initialized model, started trading loop")
	max_trades = 10  # Max number of trades
	trading_timeout = (60 * 60)  # Max runtime in seconds
	trade_count = 0
	loop_count = 0
	trading_start = dt.datetime.now()
	last_run_start = dt.datetime.now()
	last_live_datetime = None
	while True:
		try:
			debug_output(" ===== Started trade iteration " + str(loop_count))
			last_run_start = dt.datetime.now()

			# Wait for at least 1 minute to pass since last run start
			if last_live_datetime is not None:
				current_utc = dt.datetime.utcnow()
				delta = current_utc - last_live_datetime
				if delta.total_seconds() < 60:
					sleeptime = (60 - delta.total_seconds())+1
					debug_output("Sleeping for " + str(sleeptime) + " seconds")
					time.sleep(sleeptime)

			# Get results from live data
			debug_output("Waiting for live data update from broker...")
			received = False
			error_count = 0
			max_errors = 3
			attempt_start = dt.datetime.now()
			attempt_timeout = (60*5)  # Timeout after N seconds
			while not received:
				try:
					# Request data
					liveTA = LiveTA(window_size=sequenceSize+sequenceOverlap)

					# Check result time
					utc_live = liveTA.get_last_time()
					utc_now = dt.datetime.utcnow()
					utc_delta = utc_now - utc_live
					if utc_delta.total_seconds() > (60 * 10):
						debug_output("Received live data for " + utc_live.strftime("%Y-%m-%d %H:%M:%S UTC"))
						raise Exception("The UTC delta exceeds 10 minutes. Unable to retrieve recent live data")
					if last_live_datetime is None or (utc_live - last_live_datetime).total_seconds() > 0:
						debug_output("Successfully retrieved live TA data for " + utc_live.strftime("%Y-%m-%d %H:%M:%S UTC"))
						received = True
						last_live_datetime = utc_live
					else:
						time.sleep(5)
				except Exception as e:
					traceback.print_exc()
					error_count += 1
					if error_count > max_errors:
						debug_output("Caught exception while retrieving live data. terminating trade loop...")
						terminate_all()
				attempt_now = dt.datetime.now()
				attempt_delta = attempt_now - attempt_start
				if not received and attempt_delta.total_seconds() > attempt_timeout:
					debug_output("Timeout while retrieving live data. terminating trade loop...")
					terminate_all()

			# Get TA output from live data
			debug_output("Running TA on live data")
			price = liveTA.get_window_column(["bidopen"]).values
			price = price[sequenceOverlap:]
			liveTA.run_TA()
			data = liveTA.get_window_column(forex.technical_indicators).values

			# Normalize
			mean = np.mean(data, axis=0, keepdims=True)
			std = np.std(data, axis=0, keepdims=True)
			data_m = data - mean
			data = np.divide(data_m, std, out=np.zeros_like(data_m), where=std!=0)

			# Get lstm output from TA results
			debug_output("Running model on TA results")
			X = np.zeros((1, sequenceSize+sequenceOverlap, len(forex.technical_indicators)))
			X[0, :, :] = data
			Y = sess.run(y, feed_dict={x: X})

			# Interpret output
			buy_sequence = Y[0, :, 0]
			sell_sequence = Y[0, :, 1]
			# debug_output("Model output: ["+str(buy)+", "+str(sell)+"]")
			# if buy > 0 and sell < 1:
			# 	buying = True
			# if sell > 0:
			# 	selling = True

			#Debug simulate
			if True:
				fig, ax1 = plt.subplots()
				ax1.plot(price, '-k')
				ax1.plot(np.where(buy_sequence > 0)[0].tolist(), price[np.where(buy_sequence > 0)[0].tolist()].tolist(), 'rP') # buy signals
				ax1.plot(np.where(sell_sequence > 0)[0].tolist(), price[np.where(sell_sequence > 0)[0].tolist()].tolist(), 'bx') # sell signals
				ax1.set_xlabel('timestep (min)')
				ax1.set_ylabel('EUR/USD rate')
				plt.show()
				temp = None  # Breakpoint here


		except Exception as e:
			traceback.print_exc()

			# Close all open positions and terminate the program
			debug_output("Unexpected exception during trading loop. terminating trade loop...")
			terminate_all()

		loop_count += 1
		trading_now = dt.datetime.now()
		trading_delta = trading_now - trading_start
		if loop_count > 100000 or trading_delta.total_seconds() > trading_timeout:
			debug_output("Trading loop timeout. terminating trade loop...")
			terminate_all()

# Close any remaining open positions
terminate_all()
