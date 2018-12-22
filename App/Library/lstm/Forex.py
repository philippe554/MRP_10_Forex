import os
import random

import numpy as np

from App.Helpers.AccessTaDB import AccessDB


class Forex:
    directory = os.path.expanduser('~/OneDrive/0 AI Master/Research Project/HISTDATA')

    technical_indicators = ["trend_macd_diff", "trend_adx",
                            "trend_vortex_diff", "trend_trix", "trend_mass_index",
                            "trend_cci",
                            "trend_dpo", "trend_kst", "trend_kst_sig", "trend_kst_diff",
                            "trend_ichimoku_a", "trend_ichimoku_b",
                            "momentum_rsi", "momentum_uo", "momentum_tsi", "momentum_wr", "momentum_stoch",
                            "momentum_ao", "others_dr",
                            "others_dlr", "others_cr", "volatility_atr", "volatility_bbh", "volatility_bbl",
                            "volatility_bbhi",
                            "volatility_bbli", "volatility_kch", "volatility_kcl", "volatility_kchi", "volatility_kcli",
                            "volatility_dch",
                            "volatility_dcl", "volatility_dchi", "volatility_dcli"]

    def __init__(self, batch_size, sequence_size, output_size):
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.output_size = output_size

        db_access = AccessDB()
        self.db_size = db_access.get_db_size()
        self.offset = 0
        self.pd_ta = db_access.get_column(self.technical_indicators)
        self.pd_price = db_access.get_column(["barOPENBid"])
        self.restart_offset_random()

    def restart_offset_random(self):
        self.offset = int(random.random() * self.db_size)
        print("New offset set to {:,}. DB size is {:,}.".format(self.offset, self.db_size))

    def get_X(self):
        X = np.random.rand(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.random.rand(self.batch_size, self.sequence_size)

        for batch in range(self.batch_size):
            info = self.pd_ta[self.offset: (self.offset + self.sequence_size)]
            X[batch, :len(info)] = info
            price_info = self.pd_price[self.offset: (self.offset + self.sequence_size)]
            price[batch, :len(price_info)] = list(price_info.values)
            self.offset += self.sequence_size
            if self.offset > self.db_size:
                self.offset = 0

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
        profit = 0

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
                        profit += price[batch, time_step] - price_position_open

                    if not position_is_long and bull_counter >= 3:
                        profit += price_position_open - price[batch, time_step]

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

        return profit
