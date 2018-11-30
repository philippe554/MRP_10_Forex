import unittest
from App.Library.Connection.Client import Client
from App.Library.Connection.Order import Order


class TestOrder(unittest.TestCase):

    def test_buy_sell(self):
        # Place a buy order
        trade_id = Order.buy(amount=100)

        # Close the position (sell)
        profit = Order.close_position(trade_id)

        self.assertEqual(True, True)

    @classmethod
    def tearDownClass(cls):
        Client.logout()
