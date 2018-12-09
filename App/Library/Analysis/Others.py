from App.Helpers.AccessTaDB import AccessDB
from App.Helpers.LiveTA import LiveTA


class Others:

    def __init__(self, live=None):
        """
        Sets the data source
        :param live: LiveTA object containing live TA results. set to None to use historic data
        """
        if live is None:
            self.db = AccessDB()
        elif isinstance(live, LiveTA):
            self.db = live
        else:
            raise ValueError("live must be of type None or LiveTA")

    def get_dr(self, offset, window_size):
        return self.db.get_window_column(["others_dr"], offset, window_size)

    def get_dlr(self, offset, window_size):
        return self.db.get_window_column(["others_dlr"], offset, window_size)

    def get_cr(self, offset, window_size):
        return self.db.get_window_column(["others_cr"], offset, window_size)
