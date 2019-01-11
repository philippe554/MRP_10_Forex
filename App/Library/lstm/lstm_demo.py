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
from matplotlib.finance import candlestick_ohlc
from matplotlib import dates as mdates
import pandas as pd

from App.Library.Settings import settings
from App.Library.lstm.ForexDemo import ForexDemo
from App.Library.lstm.PSO import PSO
from App.Helpers.LiveTA import LiveTA

# Path to the saved model, should be a child of settings.modelPath
# pso_path = "/Saved/10-1-2351"
pso_path = "/Saved/11-1-1943"

# If simulate is set to true, the loaded model will be executed on live data using overlapping windows (similar to testing but on live data)
# If simulate is set to false, the loaded model will be used to do live trading without overlapping windows (used for actual trading)
simulate = True

print(" === Loading PSO parameters")
try:
	with open(settings.modelPath + pso_path + '/model_parameters.pkl', 'rb') as model:
		pso = pickle.load(model)
	print("     PSO loaded with parameters:")
	pso.print_hyper_parameters()
except Exception as e:
	print("     Failed to load PSO, Exiting..")
	quit()

print(" === Initializing forex class")
forex = ForexDemo(pso.sequenceSize)
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

    x = tf.stack([x[:,i:i+pso.sequenceSize,:] for i in range(pso.sequenceOverlap)], axis=1)
    check(x, [None, pso.sequenceOverlap, pso.sequenceSize, inputSize])

    # Merge the batch dimension with the overlap dimension, for tensorflow they are both batches
    x = tf.reshape(x, shape=[-1, pso.sequenceSize, inputSize])
    check(x, [None, pso.sequenceSize, inputSize])

    x = buildNN(x)
    check(x, [None, pso.sequenceSize, pso.outputSize])

    # Unfold the merge
    x = tf.reshape(x, shape=[-1, pso.sequenceOverlap, pso.sequenceSize, pso.outputSize])
    check(x, [None, pso.sequenceOverlap, pso.sequenceSize, pso.outputSize])

    x = x[:,:,-1,:]
    check(x, [None, pso.sequenceOverlap, pso.outputSize])

    return x

print(" === Building model")
if simulate:
	x = tf.placeholder("float", [None, pso.sequenceSize + pso.sequenceOverlap, inputSize])
	y = buildNNOverlap(x)
	print("     Overlap NN loaded (simulate)")
else:
	x = tf.placeholder("float", [None, pso.sequenceSize, inputSize])
	y = buildNN(x)
	print("     NN loaded (live trading)")


print(" === Initializing model variables")
variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variableSizes = [np.prod(v.get_shape().as_list()) for v in variables]
print("      Variables:", variableSizes, "Total:", np.sum(variableSizes))

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
			windowSize = pso.sequenceSize
			if simulate:
				windowSize += pso.sequenceOverlap
			while not received:
				try:
					# Request data
					liveTA = LiveTA(window_size=windowSize)

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
			if simulate:
				pricedata = liveTA.get_price_data()[pso.sequenceOverlap:]
			liveTA.run_TA()
			data = liveTA.get_window_column(forex.technical_indicators).values

			# Normalize
			mean = np.mean(data, axis=0, keepdims=True)
			std = np.std(data, axis=0, keepdims=True)
			data_m = data - mean
			data = np.divide(data_m, std, out=np.zeros_like(data_m), where=std != 0)

			# Get lstm output from TA results
			debug_output("Running model on TA results")
			X = np.zeros((1, windowSize, len(forex.technical_indicators)))
			X[0, :, :] = data
			Y = sess.run(y, feed_dict={x: X})

			# Interpret output
			buy_sequence = Y[0, :, 0]
			sell_sequence = Y[0, :, 1]

			# Debug simulate
			if simulate:
				position = 0
				balance = 0
				commission = 4  # Dollar per 100k traded
				capital = 50000
				transaction_fee = (capital / 100000) * commission
				bought = []
				sold = []
				num_buy = 0
				min_buy_signals = 1
				for i in range(pso.sequenceOverlap):
					if position == 0 and buy_sequence[i] > 0 and sell_sequence[i] == 0 and np.max(sell_sequence[i:]) > 0:
						num_buy += 1
						if num_buy >= min_buy_signals:
							# Open a new position at the current rate
							bought.append(i)
							position = pricedata['bidopen'][i]
					elif position != 0 and sell_sequence[i] > 0:
						# Close the current position
						sold.append(i)
						balance += (capital * (pricedata['bidopen'][i] - position)) - transaction_fee
						position = 0

				# Draw
				fig, ax = plt.subplots()
				pricedata.insert(0, 't', pricedata.index)
				pricedata['t'] = pd.to_datetime(pricedata['t'])
				pricedata['t'] = pricedata['t'].apply(mdates.date2num)
				ax.xaxis_date()
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
				plt.xticks(rotation=45)
				_ = candlestick_ohlc(ax, pricedata.values, width=.0003, colorup='g', alpha = .7)
				ax.plot(
					pricedata['t'][bought],
					pricedata['bidopen'][bought], 'kP', label='buy signal')
				ax.plot(
					pricedata['t'][sold],
					pricedata['bidopen'][sold], 'bx', label='sell signal')
				ymax = pricedata["bidhigh"].max()
				ymin = pricedata["bidlow"].min()
				ax.legend()
				ax.plot(pricedata['t'][np.where(buy_sequence > 0)[0].tolist()], np.full(len(np.where(buy_sequence > 0)[0].tolist()), ymin - (ymax-ymin)*.05), 'kP', label='buy signal')
				ax.plot(pricedata['t'][np.where(sell_sequence > 0)[0].tolist()], np.full(len(np.where(sell_sequence > 0)[0].tolist()), ymin - (ymax-ymin)*.1), 'bx', label='sell signal')
				ax.set_ylabel("EUR/USD")
				ax.set_xlabel(last_run_start.strftime("%Y-%m-%d"))
				ax.set_title("Gross profit/loss: " + "%.3f" % balance + "$")
				fig.autofmt_xdate()
				fig.tight_layout()
				plt.show()
			else:
				# TODO: open/close positions
				debug_output("Model output: ["+str(buy_sequence[0])+", "+str(sell_sequence[0])+"]")


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
