import random

from App.Library.lstm.ForexBase import *


class ForexSimple(ForexBase):

	def get_X_train(self):
		X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
		price = np.zeros((self.batch_size, self.sequence_size + 5))

		for batch in range(self.batch_size):
			offset = int(random.random() * (self.train_size - self.sequence_size - 5))

			X[batch, :, :] = self.TA_train[offset: (offset + self.sequence_size), :]
			price[batch, :] = self.price_train[offset: (offset + self.sequence_size) + 5, 4]

		return X, price

	def get_X_test(self, batch_size):
		X = np.zeros((batch_size, self.sequence_size, len(self.technical_indicators)))
		price = np.zeros((batch_size, self.sequence_size + 5))

		for batch in range(batch_size):
			offset = int(random.random() * (self.test_size - self.sequence_size - 5))

			X[batch, :, :] = self.TA_test[offset: (offset + self.sequence_size), :]
			price[batch, :] = self.price_test[offset: (offset + self.sequence_size) + 5, 4]

		return X, price

	def reset_stats(self):
		old_stats = self.stats.copy()
		self.stats = {
			'total': 0,
			'buy': 0,
			'sell': 0,
			'buysell': 0,
			'inactive': 0,
			'green': 0,
			'red': 0,
		}
		return old_stats

	def calculate_profit(self, price, Y):
		# Very simple: if buy signal on and next candle is green > good stuff
		batch_size = len(price)
		profit = np.zeros(batch_size)
		fee = 0.0001
		capital = 50000

		# Counts
		green = 0
		flat = 0
		red = 0
		num_buy = 0
		num_sell = 0
		buysell = 0
		inactive = 0

		for i in range(batch_size):
			self.stats['total'] += 1
			buy = np.mean(Y[i, :, 0]) > .5
			sell = np.mean(Y[i, :, 1]) > .5

			if buy and sell:
				buysell += 1
				buy = False
				sell = False
			elif buy:
				num_buy += 1
			elif sell:
				num_sell += 1
			else:
				inactive += 1

			current_rate = price[i, self.sequence_size - 1] + fee  # Add transaction fee to avoid a 'fake' profit
			next_candles = price[i, self.sequence_size+1:]
			gross_pips = (np.max(next_candles) - current_rate) + (np.min(next_candles) - current_rate)  # Use the next 5 candles
			# gross_pips = next_candles[0] - current_rate  # Use only the first next candle
			if abs(gross_pips) < fee:
				# Movement is minimal, model should do nothing
				flat += 1
				if buy or sell:
					profit[i] = -fee
				else:
					profit[i] = fee
			elif gross_pips > 0:
				green += 1
				# Next candle is green, model should have bought
				if buy:
					profit[i] = gross_pips  # This would have been the gain
				elif sell:
					profit[i] = -gross_pips  # Missed out on this
				else:
					profit[i] = 0  # Did nothing
			else:
				red += 1
				# Next candle is red, model should have sold
				if sell:
					profit[i] = -gross_pips  # Avoided this loss
				elif buy:
					profit[i] = gross_pips  # This would have been the loss
				else:
					profit[i] = 0  # Did nothing

		self.stats['green'] += green
		self.stats['red'] += red
		self.stats['buy'] += num_buy
		self.stats['sell'] += num_sell
		self.stats['buysell'] += buysell
		self.stats['inactive'] += inactive

		# Punish outputs that dont do anything
		if num_sell == 0 or num_buy == 0:
			return -1, 1

		# Check ratio
		buy_ratio = num_buy/green
		flat_ratio = inactive/flat
		sell_ratio = num_sell/red
		profit_mean = (np.mean(profit) * capital)
		batch_profit = profit_mean * buy_ratio * sell_ratio * flat_ratio

		return batch_profit, 1

	def calculate_profit_test(self, price, Y, draw):
		# TODO: Implement method
		return self.calculate_profit(price, Y)

	def evaluate_output(self, Y):
		# Mean of all buy and sell signals
		buy = np.mean(Y[0, :, 0]) > .5
		sell = np.mean(Y[0, :, 1]) > .5

		# If both are on, do nothing
		if buy and sell:
			buy = False
			sell = False

		return buy, sell
