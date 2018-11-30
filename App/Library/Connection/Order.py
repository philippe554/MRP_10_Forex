
from App.Library.Connection.Client import Client


class Order:

    @staticmethod
    def get_open_position(trade_id, connection='benchmark'):
        """
        Returns the open position for the given trade id
        :param trade_id: tradeId for the position
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :raises ValueError if the position can not be found
        """
        conn = Client.get_connection(connection, True)
        return conn.get_open_position(trade_id)

    @staticmethod
    def get_closed_position(trade_id, connection='benchmark'):
        """
        Returns the closed position for the given trade id
        :param trade_id: tradeId for the position
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :raises ValueError if the position can not be found
        """
        conn = Client.get_connection(connection, True)
        return conn.get_closed_position(trade_id)

    @staticmethod
    def buy(amount=0, stop_percentage=.95, symbol='EUR/USD', connection='benchmark'):
        """
        Opens a new buy position for the specified amount
        :param amount: amount to buy. set to 0 to go all in
        :param stop_percentage: the stop limit will be set to stop_percentage*current_market_price
        :param symbol: currency to buy
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :returns trade_id: the fxcm tradeId of the opened position
        :raises Exception
        """
        conn = Client.get_connection(connection, True)

        if conn.get_open_positions().T.size == 0:  # No open positions
            # Check account funds
            accounts = conn.get_accounts()
            usable_funds = accounts['usableMargin'][0].T
            if amount == 0:
                # Use all available funds
                buy = usable_funds
            elif amount < usable_funds:
                buy = amount
            else:
                # Insufficient funds!
                raise Exception('Insufficient funds in account')

            # Place the buy order
            order = conn.create_market_buy_order(symbol, buy)

            # Calculate the effective stop loss rate
            trade_id = order.get_tradeId()
            buy_price = order.get_buy()
            stop_rate = stop_percentage * buy_price

            # Set the stop loss rate on the active trade
            conn.change_trade_stop_limit(trade_id, is_in_pips=False, is_stop=True, rate=stop_rate)
            return trade_id
        else:
            # Positions are already opened!
            raise Exception('This account already has an open position')

    @staticmethod
    def close_position(trade_id, connection='benchmark'):
        """
        SELL: Closes the given position and returns the gross profit/loss that was made
        :param trade_id: tradeId for the position
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :returns Float: gross profit/loss after closing the position
        :raises ValueError when the position can not be found
        """
        conn = Client.get_connection(connection, True)
        position = conn.get_open_position(trade_id)
        position.close()
        profit = position.get_grossPL()
        return profit

    @staticmethod
    def close_all(connection='benchmark'):
        """
        SELL: Closes all open positions
        :param connection: specify the connection account to be used. can be either 'benchmark' or 'lstm'
        :returns Float the gross profit/loss of all closed positions. returns 0 if there are no positions to close
        """
        conn = Client.get_connection(connection, True)

        # Get all open positions
        positions = conn.get_open_positions()

        # Close all open positions and return the sum of their profit
        profit = 0
        for index, position in positions.iterrows():
            position.close()
            profit = profit + position.get_grossPL()
        return profit
