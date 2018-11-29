import unittest
from App.Library.Connection.Client import Client
from App.Library.Connection.Order import Order


class TestOrder(unittest.TestCase):

    def test_get_order_ids(self):
        # orders = Order.get_orders()
        self.assertEqual(True, True)

    def test_all_in(self):
        order = Order.all_in()
        self.assertEqual(True, True)

    @classmethod
    def tearDownClass(cls):
        Client.logout()
