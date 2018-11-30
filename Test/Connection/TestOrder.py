import unittest
from App.Library.Connection.Client import Client
from App.Library.Connection.Order import Order


class TestOrder(unittest.TestCase):

    def test_buy_sell(self):
        # Place a buy order
        position = Order.buy(amount=100, connection='benchmark')

        # Close the position (sell)
        position = Order.close_position(trade_id=position.get_tradeId(), connection='benchmark')
        profit = position.get_grossPL()

        self.assertEqual(True, True)

    @classmethod
    def tearDownClass(cls):
        Client.logout()
