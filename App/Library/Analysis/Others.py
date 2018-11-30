from App.Helpers.AccessTaDB import AccessDB


class Others:
    def __init__(self):
        self.db = AccessDB()

    def get_dr(self, offset, window_size):
        return self.db.get_window_column(["others_dr"], offset, window_size)

    def get_dlr(self, offset, window_size):
        return self.db.get_window_column(["others_dlr"], offset, window_size)

    def get_cr(self, offset, window_size):
        return self.db.get_window_column(["others_cr"], offset, window_size)
