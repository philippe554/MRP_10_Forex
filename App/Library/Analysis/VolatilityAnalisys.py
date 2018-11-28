import pandas as pd
from App.Helpers.AccessTaDB import AccessDB
import matplotlib.pyplot as plt


class VolatilityAnalisys:

    def __init__(self):
        self.db = AccessDB()

    def get_ATR(self, offset, window_size):
        """
        Average True Range (ATR) is a measure of volatility that attempts to decompose a longer trend.
        In general, ATR is higher when the volatility is higher and lower when the volatility is lower.
        https://www.investopedia.com/terms/a/atr.asp

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_atr
        """
        return self.db.get_window_column(["volatility_atr"], offset, window_size)

    def get_BollingerBands(self, offset, window_size):
        """
        The BollingerBands are a function of the moving average.
        The upper band is 2 standard deviations above the moving average; if the price crosses this band, the stock is overbought => SELL
        The lower band is 2 standard deviations below the moving average; if the price crosses this band, the stock is oversold => BUY
        The bands can also squeeze together, this indicates low volatility and thus future increased volatility
        When the bands expand, this indicates high volatility and thus future decreased volatility
        The middle band 'volatility_bbm' is omitted from this response
        https://www.investopedia.com/terms/b/bollingerbands.asp

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_bbh (upper band), volatility_bbl (lower band)
        """
        return self.db.get_window_column(["volatility_bbh", "volatility_bbl"], offset, window_size)

    def get_BollingerBandsIndicators(self, offset, window_size):
        """
        These values are derived from the bollinger bands and might perform better as a buy or sell signal.
        volatility_bbhi has a value of 1 if the candle close is higher than the upper band, otherwise has value 0
        volatility_bbli has a value of 1 if the candle close is lower than the lower band, otherwise has value 0

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_bbhi, volatility_bbli
        """
        return self.db.get_window_column(["volatility_bbhi", "volatility_bbli"], offset, window_size)

    def get_KeltnerChannel(self, offset, window_size):
        """
        Very similar to the bollinger bands in their purpose but slightly differently calcuolated.
        Basically the Keltner Channel produces much thinner bands as a volatility indicator
        https://www.investopedia.com/terms/k/keltnerchannel.asp

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_kch(upper band), volatility_kcl (lower band)
        """
        return self.db.get_window_column(["volatility_kch", "volatility_kcl"], offset, window_size)

    def get_KeltnerChannelIndicators(self, offset, window_size):
        """
        These values are derived from the keltner channel and might perform better as a buy or sell signal.
        volatility_kchi has a value of 1 if the candle close is higher than the upper keltner channel, otherwise has value 0
        volatility_kcli has a value of 1 if the candle close is lower than the lower keltner channel, otherwise has value 0

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_kchi, volatility_kcli
        """
        return self.db.get_window_column(["volatility_kchi", "volatility_kcli"], offset, window_size)

    def get_DonchianChannel(self, offset, window_size):
        """
        The donchian channel is very different from the other volatility indicators.
        The upper and lower bands are 'pushed' by the candle close price.
        E.g. if the price reaches a low after 20 days of decreasing, the lower band will be horizantal for 20 days before chaning again.
        https://www.investopedia.com/terms/d/donchianchannels.asp

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_dch(upper band), volatility_dcl (lower band)
        """
        return self.db.get_window_column(["volatility_dch", "volatility_dcl"], offset, window_size)

    def get_DonchianChannelIndicators(self, offset, window_size):
        """
        These values are derived from the donchian channels and might perform better as a buy or sell signal.
        volatility_dchi has a value of 1 if the candle close is higher than the upper donchian channel, otherwise has value 0
        volatility_dcli has a value of 1 if the candle close is lower than the lower donchian channel, otherwise has value 0

        :param offset: offset of the window
        :param window_size: size of the window
        :return: pandas with the column: volatility_dchi, volatility_dcli
        """
        return self.db.get_window_column(["volatility_dchi", "volatility_dcli"], offset, window_size)


v_an = VolatilityAnalisys()

print(v_an.get_ATR(100, 100))
