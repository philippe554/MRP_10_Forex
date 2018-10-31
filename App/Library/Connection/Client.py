import App.config as cfg
import fxcmpy


class Client:

    def __init__(self):
        self.connection = None

    def login(self):
        """
        Initialize a connection to the fxcm platform
        Returns the connection. Call con.close() to exit the connection
        """
        try:
            # Return existing connection
            if self.test_connection():
                return self.connection

            # Open a new connection
            token = cfg.forexconnect['token']
            self.connection = fxcmpy.fxcmpy(access_token=token, log_level='error')
            if self.connection.connection_status != 'established':
                raise Exception("Unable to establish forex connection")
            return self.connection
        except Exception as e:
            print(e)
            return False

    def logout(self):
        try:
            if self.test_connection():
                self.connection.close()
            self.connection = None
        except:
            return False
        return True

    def test_connection(self):
        if self.connection is not None and self.connection.connection_status == 'established':
            return True
        return False
