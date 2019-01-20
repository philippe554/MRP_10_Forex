import pickle

import numpy as np
import pandas as pd

from App.Helpers.AccessTaDB import AccessDB
from App.Library.Settings import settings


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

    def __init__(self, batch_size, sequence_size, sequence_overlap, output_size):
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.sequence_overlap = sequence_overlap
        self.output_size = output_size
        self.test_offset = 0
        self.stats = {}
        self.avgaccuracy = []
        self.bestaccuracy = []
        self.testaccuracy = []

        try:
            with open(settings.cachePath + '/cache_v2.p', 'rb') as cacheFile:
                cache = pickle.load(cacheFile)

                self.TA_train = cache["TA_train"]
                self.TA_test = cache["TA_test"]

                self.price_train = cache["price_train"]
                self.price_test = cache["price_test"]

                self.train_size = self.TA_train.shape[0]
                self.test_size = self.TA_test.shape[0]

        except:
            db_access = AccessDB()

            print("Loading TA database into RAM... (version 2)")
            TA = db_access.get_column(self.technical_indicators).values

            print("Loading price database into RAM... (version 2)")
            price = db_access.get_column(['DateTimeStamp', 'BarOPENBid', 'BarHIGHBid', 'BarLOWBid', 'BarCLOSEBid'])

            # Parse datetime
            price['DateTimeStamp'] = pd.to_datetime(price['DateTimeStamp'], format='%Y%m%d %H%M%S')
            price = price.values

            testMinutes = 60 * 24 * 200

            self.TA_train = TA[:-testMinutes, :]
            self.TA_test = TA[-testMinutes:, :]

            mean = np.mean(self.TA_train, axis=0, keepdims=True)
            std = np.std(self.TA_train, axis=0, keepdims=True)

            print(mean, std)

            self.TA_train = (self.TA_train - mean) / std
            self.TA_test = (self.TA_test - mean) / std

            self.price_train = price[:-testMinutes, :]
            self.price_test = price[-testMinutes:, :]

            self.train_size = self.TA_train.shape[0]
            self.test_size = self.TA_test.shape[0]

            cache = {}
            cache["TA_train"] = self.TA_train
            cache["TA_test"] = self.TA_test

            cache["price_train"] = self.price_train
            cache["price_test"] = self.price_test

            with open(settings.cachePath + '/cache_v2.p', 'wb') as cacheFile:
                pickle.dump(cache, cacheFile)

        print("Database loaded")

    def get_random_offset(self):
        return 0

    def reset_stats(self):
        return {}

    def get_X_train(self):
        X = np.random.rand(self.batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.random.rand(self.batch_size, self.sequence_size)

        return X, price

    def get_X_test(self, batch_size):
        X = np.random.rand(batch_size, self.sequence_size, len(self.technical_indicators))
        price = np.random.rand(batch_size, self.sequence_size)

        return X, price

    def evaluate_output(self, Y):
        buy = False
        sell = False

        return buy, sell

    def calculate_profit(self, price, Y):
        return 0

    def calculate_profit_test(self, price, Y, draw):
        return 0

    def restart_offset_random(self):
        pass
