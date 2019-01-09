import random
import numpy as np
import matplotlib.pyplot as plt

from App.Library.lstm.ForexBase import *


class ForexOverlap(ForexBase):
	def get_X_train(self):
		X = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap, len(self.technical_indicators)))
		price = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap))

		for batch in range(self.batch_size):
			offset = int(random.random() * (self.train_size - (self.sequence_size + self.sequence_overlap)))

			X[batch, :, :] = self.TA_train[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]
			price[batch, :] = self.price_train[offset: (offset + (self.sequence_size + self.sequence_overlap)), 0]

		return X, price

	def get_X_test(self):
		X = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap, len(self.technical_indicators)))
		price = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap))

		for batch in range(self.batch_size):
			offset = int(random.random() * (self.test_size - (self.sequence_size + self.sequence_overlap)))

			X[batch, :, :] = self.TA_test[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]
			price[batch, :] = self.price_test[offset: (offset + (self.sequence_size + self.sequence_overlap)), 0]

		return X, price

	def calculate_profit(self, price, Y):
		# Basic idea: simulate profit that would be earned in a live environment
		# Punish outputs that do not trade very often

		transaction_fee = 0.00001
		capital = 50000
		position_counts = np.zeros(self.batch_size)
		balance = np.zeros(self.batch_size)

		for batch in range(self.batch_size):
			price_overlap = price[batch, len(price[batch])-self.sequence_overlap:]
			position = 0
			buy_sequence = Y[batch, :, 0]
			sell_sequence = Y[batch, :, 1]

			# Determine the price movement in the window
			diffs = np.diff(price_overlap)
			movement_up = np.sum(diffs[diffs>0])
			movement_down = np.sum(diffs[diffs<0])

			bought = []
			sold = []
			for i in range(self.sequence_overlap):
				if position == 0 and buy_sequence[i] > 0 and sell_sequence[i] == 0 and np.max(sell_sequence[i:]) > 0:
					# Open a new position at the current rate
					bought.append(i)
					position = price_overlap[i]
					position_counts[batch] += 1
				elif position != 0 and sell_sequence[i] > 0:
					# Close the current position
					sold.append(i)
					balance[batch] += capital * ((price_overlap[i] - position) - transaction_fee)
					position = 0

			# Debug plots
			if False and balance[batch] > 0:
				fig, ax1 = plt.subplots()
				ax1.set_title("Gross profit/loss: " + "%.5f" % balance[batch] + "$")
				# ax2 = ax1.twinx()
				ax1.plot(price_overlap, '-k')
				# ax1.plot(np.where(buy_sequence > 0)[0].tolist(), price_overlap[np.where(buy_sequence > 0)[0].tolist()].tolist(), 'rP') # buy signals
				# ax1.plot(np.where(sell_sequence > 0)[0].tolist(), price_overlap[np.where(sell_sequence > 0)[0].tolist()].tolist(), 'bx') # sell signals
				ax1.plot(bought, price_overlap[bought].tolist(), 'rP') # buy signals
				ax1.plot(sold, price_overlap[sold].tolist(), 'bx') # sell signals
				ax1.set_xlabel('timestep (min)')
				ax1.set_ylabel('EUR/USD rate')
				plt.show()

		return np.mean(balance), np.sum(position_counts) / self.batch_size
