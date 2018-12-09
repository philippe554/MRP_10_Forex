from App.Helpers.AccessTaDB import AccessDB
from App.Helpers.LiveTA import LiveTA


class MomentumAnalyses:

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

    def get_RSI(self, offset, window_size):
        """Relative Strength Index (RSI)
            Compares the magnitude of recent gains and losses over a specified time
            period to measure speed and change of price movements of a security. It is
            primarily used to attempt to identify overbought or oversold conditions in
            the trading of an asset.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_rsi
        """
        return self.db.get_window_column(["momentum_rsi"], offset, window_size)

    def get_TSI(self, offset, window_size):
        """True strength index (TSI)
            Shows both trend direction and overbought/oversold conditions.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_tsi
        """
        return self.db.get_window_column(["momentum_tsi"], offset, window_size)

    def get_UO(self, offset, window_size):
        """Ultimate Oscillator
            Larry Williams' (1976) signal, a momentum oscillator designed to capture momentum
            across three different timeframes.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_uo
        """
        return self.db.get_window_column(["momentum_uo"], offset, window_size)

    def get_Stoch(self, offset, window_size):
        """Stochastic Oscillator
            Developed in the late 1950s by George Lane. The stochastic
            oscillator presents the location of the closing price of a
            stock in relation to the high and low range of the price
            of a stock over a period of time, typically a 14-day period.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_stoch
        """

        return self.db.get_window_column(["momentum_stoch"], offset, window_size)

    def get_Stoch_Signal(self, offset, window_size):
        """Stochastic Oscillator Signal
            Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_stoch_signal
        """
        return self.db.get_window_column(["momentum_stoch_signal"], offset, window_size)

    def get_WR(self, offset, window_size):
        """Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
            The Fast Stochastic Oscillator and Williams %R produce the exact same lines,
            only the scaling is different. Williams %R oscillates from 0 to -100.

            :param offset: offset of the window
            :param window_size: size of the window
            :return: pandas with the column: momentum_wr
        """
        return self.db.get_window_column(["momentum_wr"], offset, window_size)

    def get_AO(self, offset, window_size):
        """
        The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a
        34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated
        using closing price but rather each bar's midpoints. AO is generally used to affirm trends or to anticipate
        possible reversals.

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: momentum_ao
        """

        return self.db.get_window_column(["momentum_ao"], offset, window_size)

# mo_an = MomentumAnalyses()
# print(mo_an.get_RSI(100, 100))
