
from App.Library.Connection.Client import Client


class Order:

    @staticmethod
    def get_orders():
        return Client.get_connection().get_orders()

    #TODO: add buy/sell methods
