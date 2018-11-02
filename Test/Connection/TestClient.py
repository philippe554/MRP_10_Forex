import unittest
from App.Library.Connection.Client import Client


class TestClient(unittest.TestCase):

    def test_login(self):
        conn = Client.get_connection()
        self.assertEqual(conn.connection_status, 'established')

        conn = Client.get_connection()
        self.assertEqual(conn.connection_status, 'established')

    @classmethod
    def tearDownClass(cls):
        Client.logout()
