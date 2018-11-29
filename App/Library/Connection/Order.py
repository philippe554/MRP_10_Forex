
from App.Library.Connection.Client import Client


class Order:

    @staticmethod
    def get_orders():
        return Client.get_connection().get_orders()

    @staticmethod
    def all_in():
        """
        BUY: Goes all in with the entire portfolio, unless already in this position
        """
        conn = Client.get_connection()

        if conn.get_open_positions().T.size == 0:
            # No open positions, place a stop loss order with the full portfolio
            accounts = conn.get_accounts()
            usable_funds = accounts['balance'][0].T

            # Place a buy order
            trade = conn.create_market_buy_order('EUR/USD', 1000)

            # Set a stop loss rate
            conn.change_trade_stop_limit(trade['tradeId'], is_in_pips=False, is_stop=True, rate=115)
            return True
        else:
            # Positions are already opened!
            return True

    @staticmethod
    def sell_all():
        """
        SELL: Closes all open positions
        """
        return Client.get_connection().close_all()
