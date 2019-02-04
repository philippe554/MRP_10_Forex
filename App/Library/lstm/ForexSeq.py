import random

from App.Library.lstm.ForexBase import *


class ForexSeq(ForexBase):

	def get_random_offset(self):
		return int(random.random() * (self.train_size - self.sequence_size))

	def get_X_train(self):
		X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
		price = np.zeros((self.batch_size, self.sequence_size))

		for batch in range(self.batch_size):
			if self.offset > self.train_size - self.sequence_size:
				self.offset = 0

			X[batch, :] = self.TA_train[self.offset: (self.offset + self.sequence_size)]
			price[batch, :] = self.price_train[self.offset: (self.offset + self.sequence_size), 1]
			self.offset += self.sequence_size

		return X, price

	def get_X_test(self, batch_size):
		X = np.zeros((1, self.sequence_size, len(self.technical_indicators)))
		price = np.zeros((1, self.sequence_size))

		X[0, :] = self.TA_test[self.test_offset: (self.test_offset + self.sequence_size)]
		price[0, :] = self.price_test[self.test_offset: (self.test_offset + self.sequence_size), 1]
		self.test_offset += 1

		return X, price

	def reset_stats(self):
		return {}

	def calculate_profit(self, price, Y):
		"""
				Calculate the profit of this period
				:param price: price history
				:param Y: output of the LSTM being for each time-stamp a 2-dimension array. The first position is considered
				as a bullish indicator and the bottom as a bear indicator
				:return:
				"""
		batch_size = len(price)
		assert list(np.shape(price)) == [batch_size, self.sequence_size]
		assert list(np.shape(Y)) == [batch_size, self.sequence_size, self.output_size]
		output = Y.round()

		"""
		The technique applied is:
		- long position: after 5 timestamps with bull indicator ON and bear OFF
		- short position: after 5 timestamps with bull indicator OFF and bear ON
		- close position: after 3 following timestamp with mixed indicator or opposite of position 
			(if we opened a long position and we receive three times a bear indicator, 
			 or we opened a short position and we receive three times a bull indicator)
		"""
		bull_counter = 0
		bear_counter = 0
		position_open = False
		position_is_long = False
		price_position_open = 0
		start_money = 10000
		money = start_money
		n_positions = 0

		for batch in range(batch_size):
			for time_step in range(self.sequence_size):
				bull_bear_indicators = output[batch, time_step]

				if bull_bear_indicators[0] == 1:  # Bull indicator is ON
					bull_counter += 1
				else:
					bull_counter = 0
				if bull_bear_indicators[1] == 1:  # Bear indicator is ON
					bear_counter += 1
				else:
					bear_counter = 0

				if position_open:
					if position_is_long and bear_counter >= 3:
						money += money * (price[batch, time_step] - price_position_open)
						money *= 0.9995
						position_open = False
						n_positions += 1

					if not position_is_long and bull_counter >= 3:
						money += money * (price_position_open - price[batch, time_step])
						money *= 0.9995
						position_open = False
						n_positions += 1

				else:
					if bull_counter >= 5:
						bull_counter = 0
						if bear_counter > 0:
							bear_counter = 0
						else:
							position_open = True
							position_is_long = True
							price_position_open = price[batch, time_step]

					if bear_counter >= 5:
						bear_counter = 0
						if bull_counter > 0:
							bull_counter = 0
						else:
							position_open = True
							position_is_long = False
							price_position_open = price[batch, time_step]
		profit = money - start_money
		if profit > 0:
			profit = n_positions * (money - start_money)

		return profit, n_positions

	def restart_offset_random(self):
		self.offset = int(random.random() * (self.train_size - self.sequence_size))
		print("New offset set to {:,}. DB size is {:,}.".format(self.offset, self.train_size))

		def evaluate_output(self, Y):
			# TODO: Implement, just taking last column for now
			buy = False
			sell = False

			# Check last column
			if Y[0, -1, 0] > 0:
				buy = True
			if Y[0, -1, 1] > 0:
				sell = True

			return buy, sell

	def calculate_profit_test(self, price, Y, draw):
		assert list(np.shape(price)) == [1, self.sequence_size]
		assert list(np.shape(Y)) == [1, self.sequence_size, self.output_size]
		bull_bear_indicators = Y.round()[0, -1, :]

		self.initialize_variables()

		if bull_bear_indicators[0] == 1:  # Bull indicator is ON
			self.bull_counter += 1
		else:
			self.bull_counter = 0

		if bull_bear_indicators[1] == 1:  # Bear indicator is ON
			self.bear_counter += 1
		else:
			self.bear_counter = 0

		if self.position_open:
			if self.position_is_long and self.bear_counter >= 3:
				profit = self.money * (price[0, -1] - self.price_position_open)
				self.money += profit
				fee = self.money * 0.0005
				self.money *= 0.9995
				self.position_open = False
				self.n_positions += 1
				print("Long position closed with a profit of", round(profit, 2), "paid fee of", round(fee, 2), "won",
					  round(profit - fee, 2))

			if not self.position_is_long and self.bull_counter >= 3:
				profit = self.money * (self.price_position_open - price[0, -1])
				self.money += profit
				fee = self.money * 0.0005
				self.money *= 0.9995
				self.position_open = False
				self.n_positions += 1
				print("Short position closed with a profit of", round(profit, 2), "paid fee of", round(fee, 2), "won",
					  round(profit - fee, 2))

		else:
			if self.bull_counter >= 5:
				self.bull_counter = 0
				if self.bear_counter > 0:
					self.bear_counter = 0
				else:
					self.position_open = True
					self.position_is_long = True
					self.price_position_open = price[0, -1]

			if self.bear_counter >= 5:
				self.bear_counter = 0
				if self.bull_counter > 0:
					self.bull_counter = 0
				else:
					self.position_open = True
					self.position_is_long = False
					self.price_position_open = price[0, -1]

		return self

	def initialize_variables(self):
		try:
			self.money, self.position_is_long, self.position_open, self.price_position_open, self.bull_counter
			self.bear_counter, self.n_positions
		except AttributeError:
			self.money = 10000
			self.position_is_long = False
			self.position_open = False
			self.price_position_open = 0
			self.bull_counter = 1
			self.bear_counter = 1
			self.n_positions = 0
