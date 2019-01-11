import os
import random
import pickle

import numpy as np

from App.Library.Settings import settings
from App.Helpers.AccessTaDB import AccessDB

class ForexDemo:
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

    def __init__(self,sequence_size):
        self.sequence_size = sequence_size

    def get_random_offset(self):
        return 0

    def get_X_train(self):
        return 0

    def get_X_test(self):
        return 0

    def calculate_profit(self, price, Y):
        return 0

    def restart_offset_random(self):
        pass