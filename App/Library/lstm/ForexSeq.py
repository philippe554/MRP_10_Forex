import random
import numpy as np

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

            info = self.TA_train[self.offset : (self.offset + self.sequence_size)]
            X[batch , :] = info
            price[batch , :] = self.price_train[self.offset : (self.offset + self.sequence_size), 0]
            self.offset += self.sequence_size


        return X, price

    def get_X_test(self):
        assert False
        X = np.zeros((self.batch_size, self.sequence_size, len(self.technical_indicators)))
        price = np.zeros((self.batch_size, self.sequence_size))

        return X, price

    def calculate_profit(self, price, Y):
        """
                Calculate the profit of this period
                :param price: price history
                :param Y: output of the LSTM being for each time-stamp a 2-dimension array. The first position is considered
                as a bullish indicator and the bottom as a bear indicator
                :return:
                """
        assert list(np.shape(price)) == [self.batch_size, self.sequence_size]
        assert list(np.shape(Y)) == [self.batch_size, self.sequence_size, self.output_size]
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
        money = 10000
        n_positions = 0

        for batch in range(self.batch_size):
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
                        money *= 0.99
                        money += money * (price[batch, time_step] - price_position_open)
                        position_open = False
                        n_positions += 1

                    if not position_is_long and bull_counter >= 3:
                        money *= 0.99
                        money += money * (price_position_open - price[batch, time_step])
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

        return money - 10000, n_positions

    def restart_offset_random(self):
        self.offset = int(random.random() * (self.train_size - self.sequence_size))
        print("New offset set to {:,}. DB size is {:,}.".format(self.offset, self.train_size))