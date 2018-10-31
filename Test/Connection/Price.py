import unittest
from App.Library.Connection.Client import Client
from App.Library.Connection.Price import Price
from App.Library.Enum.Period import Period

import datetime

class TestForexConnection(unittest.TestCase):

    def test_get_last_n_candles(self):
        candles = Price.get_last_n_candles(instrument='EUR/USD', period=Period.MINUTE_1[0], n=30)
        self.assertEqual(candles.shape, (30,9))

    def test_get_candles_from_interval(self):
        start = datetime.datetime(2018, 6, 20)
        stop = datetime.datetime(2018, 10, 20)
        candles = Price.get_candles_from_interval(instrument='EUR/USD', period=Period.DAY_1[0], start=start, stop=stop)
        self.assertEqual(candles.shape, (104,9))

    @classmethod
    def tearDownClass(cls):
        Client.logout()
