from App.Library.Connection.Price import Price
from ta import *
from datetime import datetime as dt

from App.Library.Enum.Period import Period


class LiveTA:

    def __init__(self, window_size=240):
        self.window_padding = 300  # Window padding is added to the start of the window for those indicators that need a run-up period
        self.data = Price.get_last_n_candles(instrument='EUR/USD', period=Period.MINUTE_1[0], n=(self.window_padding+window_size))

    def get_window_column(self, columns, offset=None, window_size=None):
        """
        Returns the specified columns from the TA results. ignores offset and window size as they have to be set during initialization
        :param columns: list of column names e.g. ['volatility_kcc', 'volatility_kch']
        :param offset: Unused
        :param window_size: set during initialization
        :return: panda dataframe with the specified columns
        """
        return self.data[columns][self.window_padding:]

    def get_last_time(self):
        last_index = self.data.index[-1]
        return last_index.to_pydatetime()

    def run_TA(self):
        """
        Runs technical analysis on the data frame
        """
        open = "bidopen"
        high = "bidhigh"
        low = "bidlow"
        close = "bidclose"
        fillna = True
        df = self.data
        df = add_volatility_ta(df, high, low, close, fillna=fillna)
        df = add_trend_ta(df, high, low, close, fillna=fillna)
        df = add_others_ta(df, close, fillna=fillna)
        df['momentum_rsi'] = rsi(df[close], n=14, fillna=fillna)
        df['momentum_tsi'] = tsi(df[close], r=25, s=13, fillna=fillna)
        df['momentum_uo'] = uo(df[high], df[low], df[close], fillna=fillna)
        df['momentum_stoch'] = stoch(df[high], df[low], df[close], fillna=fillna)
        df['momentum_stoch_signal'] = stoch_signal(df[high], df[low], df[close], fillna=fillna)
        df['momentum_wr'] = wr(df[high], df[low], df[close], fillna=fillna)
        df['momentum_ao'] = ao(df[high], df[low], fillna=fillna)
        self.data = df
        return self.data