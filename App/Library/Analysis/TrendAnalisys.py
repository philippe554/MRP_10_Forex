import pandas as pd
from App.Helpers.AccessTaDB import AccessDB
import matplotlib.pyplot as plt
from App.Helpers.LiveTA import LiveTA


class TrenAnalisys:

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

    def get_MACD(self, offset, window_size):
        """
        Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator. The signal of a change
        of trend is given with the crossover of the values of MACD, therefore, we retrieve the difference between them.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_macd_diff
        """
        return self.db.get_window_column(["trend_macd_diff"], offset, window_size)

    def get_ADX(self, offset, window_size):
        """
        Get the Average Directional Index, a momentum strength indicators. The ADX identifies a strong positive trend
        when the ADX is over 25 and a weak trend when the ADX is below 20.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_adx
        """
        return self.db.get_window_column(["trend_adx"], offset, window_size)

    def get_VI(self, offset, window_size):
        """
        Get the vortex indicator values. It is composed of 2 lines that show both positive (VI +) and negative (VI -)
        trend movement. The signal of a change of trend is given when the lines cross, therefore, we retrieve the
        difference between them.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_vortex_diff
        """
        return self.db.get_window_column(["trend_vortex_diff"], offset, window_size)

    def get_TRIX(self, offset, window_size):
        """
        Get the TRIX values. It is designed to filter out price movements that are considered insignificant or
        unimportant.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_trix
        """
        return self.db.get_window_column(["trend_trix"], offset, window_size)

    def get_MI(self, offset, window_size):
        """
        Get Mass index. This indicator examines the range between high and low stock prices over a period of time.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_mass_index
        """
        return self.db.get_window_column(["trend_mass_index"], offset, window_size)

    def get_CCI(self, offset, window_size):
        """
        Get the Commodity Channel Index. Momentum-based technical trading tool used most often to help determine when
        an investment vehicle is reaching a condition of being overbought or oversold.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_cci
        """
        return self.db.get_window_column(["trend_cci"], offset, window_size)

    def get_DPO(self, offset, window_size):
        """
        Get the Detrended Price Oscillator. Oscillator that strips out price trends in an effort to estimate the length
        of price cycles from peak to peak, or trough to trough.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: trend_dpo
        """
        return self.db.get_window_column(["trend_dpo"], offset, window_size)


    def get_KST(self, offset, window_size):
        """
        In Know Sure Thing indicator it is important not only the crossover but also the trend of the indicator itself.
        For this reason, the three values are needed.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the columns: trend_kst, trend_kst_sig, trend_kst_diff
        """
        return self.db.get_window_column(["trend_kst, trend_kst_sig, trend_kst_diff"], offset, window_size)

    def get_Ichimoku(self, offset, window_size):
        """
        Get the values of the Ichimoku cloud. Shows support and resistance, and momentum and trend directions.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the columns: trend_ichimoku_a, trend_ichimoku_b
        """
        return self.db.get_window_column(["trend_ichimoku_a, trend_ichimoku_b"], offset, window_size)


# t_an = TrenAnalisys()
#
# print(t_an.get_Ichimoku(100, 100))
# print(t_an.get_Ichimoku(0,100))
