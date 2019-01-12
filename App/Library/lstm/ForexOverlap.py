import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib import dates as mdates

from App.Library.lstm.ForexBase import *


class ForexOverlap(ForexBase):
	def get_X_train(self):
		X = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap, len(self.technical_indicators)))
		price = np.zeros((self.batch_size, self.sequence_size + self.sequence_overlap, 5), dtype=object)

		for batch in range(self.batch_size):
			offset = int(random.random() * (self.train_size - (self.sequence_size + self.sequence_overlap)))

			X[batch, :, :] = self.TA_train[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]
			price[batch, :, :] = self.price_train[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]

		return X, price

	def get_X_test(self, batch_size):
		X = np.zeros((batch_size, self.sequence_size + self.sequence_overlap, len(self.technical_indicators)))
		price = np.zeros((batch_size, self.sequence_size + self.sequence_overlap, 5), dtype=object)

		for batch in range(batch_size):
			offset = int(random.random() * (self.test_size - (self.sequence_size + self.sequence_overlap)))

			X[batch, :, :] = self.TA_test[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]
			price[batch, :, :] = self.price_test[offset: (offset + (self.sequence_size + self.sequence_overlap)), :]

		return X, price

	def reset_stats(self):
		old_stats = self.stats.copy()
		self.stats = {
			'total': 0,
			'numNonTrade': 0,
			'buyAlwaysOn': 0,
			'sellAlwaysOn': 0,
			'buyNeverOn': 0,
			'sellNeverOn': 0
		}
		return old_stats

	def calculate_profit(self, price, Y, test=False, draw=False):
		# Basic idea: simulate profit that would be earned in a live environment
		# Punish outputs that do not trade very often
		batch_size = len(price)

		drawn = False
		commission = 4  # Dollar per 100k traded
		capital = 50000
		transaction_fee = (capital / 100000) * commission
		min_buy_signals = 1  # Wait for this number of signals before buying

		position_counts = np.zeros(batch_size)
		balance = np.zeros(batch_size)

		for batch in range(batch_size):
			self.stats['total'] += 1
			price_overlap = price[batch, len(price[batch]) - self.sequence_overlap:]
			position = 0
			buy_sequence = Y[batch, :, 0]
			sell_sequence = Y[batch, :, 1]

			price_max = price_overlap[:, 2].max()  # high
			price_min = price_overlap[:, 3].min()  # low
			price_diff = price_max - price_min

			bought = []
			sold = []
			num_buy = 0
			for i in range(self.sequence_overlap):
				if position == 0 and buy_sequence[i] > 0 and sell_sequence[i] == 0 and np.max(sell_sequence[i:]) > 0:
					num_buy += 1
					if num_buy >= min_buy_signals:
						# Open a new position at the current rate (bar close)
						bought.append(i)
						position = price_overlap[i, 4]
						position_counts[batch] += 1
				elif position != 0 and sell_sequence[i] > 0:
					# Close the current position (bar close)
					num_buy = 0
					sold.append(i)
					balance[batch] += (capital * (price_overlap[i, 4] - position)) - transaction_fee
					position = 0
				elif buy_sequence[i] == 0:
					num_buy = 0

			gross = balance[batch]

			# Add some heuristics
			# profit/loss should be a function of price movement (e.g. profit is more impressive if there is little movement)
			diffs = np.diff(price_overlap[:, 4])
			movement_up = np.sum(diffs[diffs > 0])
			# movement_down = np.sum(diffs[diffs < 0])
			max_profit = (capital * movement_up)
			if balance[batch] > max_profit:
				print("balance exceeds max profit?")
			balance[batch] = (balance[batch] / max(1, balance[batch], max_profit)) * 150

			# More trades = better
			balance[batch] *= position_counts[batch]

			# Force some trades by punishing non-trade outputs
			if position_counts[batch] == 0:
				self.stats['numNonTrade'] += 1
				balance[batch] = -(.2 * max_profit)

			# Sell sequence is always on or always off
			if np.min(sell_sequence) > 0:
				self.stats['sellAlwaysOn'] += 1
				balance[batch] = -(.7 * max_profit)
			if np.max(sell_sequence) < 1:
				self.stats['sellNeverOn'] += 1
				balance[batch] = -(.5 * max_profit)

			# Buy sequence is always on or always off
			if np.min(buy_sequence) > 0:
				self.stats['buyAlwaysOn'] += 1
				balance[batch] = -(.7 * max_profit)
			if np.max(buy_sequence) < 1:
				self.stats['buyNeverOn'] += 1
				balance[batch] = -(.5 * max_profit)

			# Debug plots
			if draw and not drawn:
				# if True and balance[batch] > 0:
				drawn = True
				if isinstance(price_overlap[-1, 0], float):
					day_label = mdates.num2date(price_overlap[-1, 0]).strftime("%Y-%m-%d") + " UTC"
				else:
					day_label = price_overlap[-1, 0].strftime("%Y-%m-%d") + " UTC"
					for tk in range(self.sequence_overlap):
						price_overlap[tk, 0] = mdates.date2num(price_overlap[tk, 0])

				# Draw
				fig, ax = plt.subplots()
				ax.xaxis_date()
				ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
				plt.xticks(rotation=45)
				_ = candlestick_ohlc(ax, price_overlap, width=.0005, colorup='#64a053', colordown='#cc4b4b', alpha=1)

				# Signals
				ax.plot(
					price_overlap[bought, 0].tolist(),
					(price_overlap[bought, 2] + price_diff * .05).tolist(), 'k+', label='buy signal')
				ax.plot(
					price_overlap[sold, 0].tolist(),
					(price_overlap[sold, 2] + price_diff * .05).tolist(), 'bx', label='sell signal')
				ax.legend()
				ax.plot(price_overlap[np.where(buy_sequence > 0)[0].tolist(), 0].tolist(),
						np.full(len(np.where(buy_sequence > 0)[0].tolist()), price_min - price_diff * .05).tolist(),
						'k+',
						label='buy signal')
				ax.plot(price_overlap[np.where(sell_sequence > 0)[0].tolist(), 0].tolist(),
						np.full(len(np.where(sell_sequence > 0)[0].tolist()), price_min - price_diff * .1).tolist(),
						'bx',
						label='sell signal')

				ax.set_ylabel("EUR/USD")
				ax.set_xlabel(day_label)
				ax.set_title("Gross profit: " + "%.3f" % gross + "$ [max profit: "+"%.3f" % max_profit+", cost: " + "%.3f" % -balance[batch] + "]")
				fig.autofmt_xdate()
				fig.tight_layout()
				plt.show()
				temp = None  # Breakpoint here

			if test:
				# Ignore heuristics during test to get real profit
				balance[batch] = gross

		return np.mean(balance), np.sum(position_counts) / batch_size

	def calculate_profit_test(self, price, Y, draw):
		return self.calculate_profit(price, Y, test=True, draw=draw)
