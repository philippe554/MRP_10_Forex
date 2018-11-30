import App.config as cfg
import fxcmpy


class Client:
    connection = None

    def __init__(self):
        self.connection = None

    @staticmethod
    def get_connection(connection_type='benchmark', force_new=False):
        try:
            if not force_new and Client.test_connection():
                # Return existing connection
                return Client.connection
            else:
                # Open a new connection
                token = cfg.forexconnect[connection_type]['token']
                connection = fxcmpy.fxcmpy(access_token=token, log_level='error')
                if not connection.is_connected():
                    raise Exception("Unable to establish forex connection")
                Client.connection = connection
                return Client.connection
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def logout():
        try:
            if Client.test_connection():
                Client.connection.close()
            Client.connection = None
        except:
            return False
        return True

    @staticmethod
    def test_connection():
        if Client.connection is not None and Client.connection.is_connected():
            return True
        return False
