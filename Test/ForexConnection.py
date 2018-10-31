import unittest
from App.Library.Connection.Client import Client


class TestForexConnection(unittest.TestCase):

    def test_login(self):
        fx = Client()
        login = fx.login()
        self.assertEqual(login.connection_status, 'established')

        login = fx.login()
        self.assertEqual(login.connection_status, 'established')

        fx.logout()
