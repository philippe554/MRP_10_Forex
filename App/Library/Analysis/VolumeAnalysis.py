import pandas as pd
from App.Helpers.AccessTaDB import AccessDB
import matplotlib.pyplot as plt
from App.Helpers.LiveTA import LiveTA


class VolumeAnalysis:

    def __init__(self, live=None):
        """
        Sets the data source
        :param live: LiveTA object containing live TA results. set to None to use historic data
        """
        if live is None:
            self.db = AccessDB()
        elif isinstance(live, LiveTA):
            self.db = live
        else:
            raise ValueError("live must be of type None or LiveTA")

    def get_ADI(self, offset, window_size):
        """
        Accumulation distribution indicator (ADI) The ADI is an indicator that helps predict reversals

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_adi
        """
        return self.db.get_window_column(["volume_adi"], offset, window_size)

    def get_OBV(self, offset, window_size):
        """
        On-balance volume is a technical analysis indicator intended to relate price and volume in the stock marke

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_obv
        """
        return self.db.get_window_column(["volume_obv"], offset, window_size)

    def get_OBV_mean(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_obvm
        """
        return self.db.get_window_column(["volume_obvm"], offset, window_size)

    def get_CMF(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_cmf
        """
        return self.db.get_window_column(["volume_cmf"], offset, window_size)

    def get_FI(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_fi
        """
        return self.db.get_window_column(["volume_fi"], offset, window_size)

    def get_EMV(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_em'
        """
        return self.db.get_window_column(["volume_em"], offset, window_size)

    def get_VPT(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_vpt'
        """
        return self.db.get_window_column(["volume_vpt"], offset, window_size)

    def get_NVI(self, offset, window_size):
        """
        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volume_nvi'
        """
        return self.db.get_window_column(["volume_nvi"], offset, window_size)


# v_analysis = VolumeAnalysis()
# print(v_analysis.get_ADI(100, 100))
