import App.config as cfg
from App.Library.Connection.Client import Client
from App.Library.Enum.Period import Period

import datetime


class Price:

    @staticmethod
    def get_last_n_candles(instrument=cfg.instrument, period=Period.MINUTE_1[0], n=60, connection='benchmark'):
        """
        Returns the last n candles of the given instrument and period
        :param instrument: Currency pair e.g. 'EUR/USD'
        :param period: Candle time interval e.g. 'm1'
        :param n: Number of candles to retrieve
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :return DataFrame: The last n candles
        """
        return Client.get_connection(connection, False).get_candles(instrument, period=period, number=n)

    @staticmethod
    def get_candles_from_interval(
            instrument=cfg.instrument,
            period=Period.DAY_1[0],
            start=datetime.datetime(2018,6,20),
            stop=datetime.datetime(2018,10,20),
            connection='benchmark'):
        """
        Returns all candles in the given interval
        https://www.fxcm.com/fxcmpy/02_historical_data.html#Time-Windows
        :param instrument: Currency pair e.g. 'EUR/USD'
        :param period: Candle time interval e.g. 'm1'
        :param start: datetime start date
        :param stop: datetime stop date
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :return DataFrame: The candles in the given time window
        """
        return Client.get_connection(connection, False).get_candles(instrument, period=period, start=start, stop=stop)
