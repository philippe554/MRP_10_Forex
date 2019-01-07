import os
import random

import numpy as np

from App.Helpers.AccessTaDB import AccessDB

class ForexBase:
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

        print("Loading database into RAM...")

        db_access = AccessDB()

        TA = db_access.get_column(self.technical_indicators).values
        price = db_access.get_column(["barOPENBid"]).values

        testMinutes = 60*24*200

        self.TA_train = TA[:-testMinutes,:]
        self.TA_test = TA[-testMinutes:, :]

        self.price_train = price[:-testMinutes, :]
        self.price_test = price[-testMinutes:, :]

        self.train_size = self.TA_train.shape[0]
        self.test_size = self.TA_test.shape[0]

        print("Database loaded")

    def get_random_offset(self):
        return 0

    def get_X_train(self):
        X = np.random.rand(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.random.rand(self.batch_size, self.sequence_size)

        return X, price

    def get_X_test(self):
        X = np.random.rand(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.random.rand(self.batch_size, self.sequence_size)

        return X, price

    def calculate_profit(self, price, Y):
        return 0

    def restart_offset_random(self):
        pass