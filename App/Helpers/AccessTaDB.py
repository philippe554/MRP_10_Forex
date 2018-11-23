import pandas as pd
import sqlite3
import os

inputFile = os.path.expanduser('~/Google Drive/Group 10_ Forecast the FOREX market/Data/price_hist_ta.db')


class AccessDB:

    def __init__(self, offset=0):
        self.conn = sqlite3.connect(inputFile)
        self.offset = offset
        self.size = 300000

    def get_column(self, columns):
        """
        Get columns from the database with all the technical indicators

        :param columns: Array of strings with the name of the columns to extract
        :return: Pandas object with the required columns
        """
        done = False
        new_offset = self.offset
        while not done:
            # print("next batch: " + new_offset.__str__())
            conn = sqlite3.connect(inputFile)
            new_value = pd.read_sql_query("SELECT " + ','.join(columns) + " FROM price_hist LIMIT "
                                          + self.size.__str__() + " OFFSET " + new_offset.__str__(), conn)

            if self.offset is new_offset:
                df = new_value
            else:
                df = df.append(new_value, ignore_index=True)

            if new_value.size < self.size:
                done = True

            new_offset += self.size
        return df

    def get_end_window_column(self, columns, window_size):
        """
        Get columns from the database with all the technical indicators only for a window at the end of the DB

        :param columns: Array of strings with the name of the columns to extract
        :param window_size: Size of the window to retrieve
        :return: Pandas object with the required columns
        """
        done = False

        conn = sqlite3.connect(inputFile)
        n_rows = pd.read_sql_query("SELECT Count(*) FROM price_hist", conn)

        first_offset = n_rows.values[0][0] - window_size
        new_offset = first_offset

        while not done:
            # print("next batch: " + new_offset.__str__())
            conn = sqlite3.connect(inputFile)
            new_value = pd.read_sql_query("SELECT " + ','.join(columns) + " FROM price_hist LIMIT "
                                          + self.size.__str__() + " OFFSET " + new_offset.__str__(), conn)

            if new_offset is first_offset:
                df = new_value
            else:
                df = df.append(new_value, ignore_index=True)

            if new_value.size < self.size:
                done = True

            new_offset += self.size

        return df

    def get_window_column(self, columns, offset, window_size):
        """
        Get columns from the database with all the technical indicators only for a window

        :param columns: Array of strings with the name of the columns to extract
        :param offset: Start point of the window to retrieve
        :param window_size: Size of the window to retrieve
        :return: Pandas object with the required columns
        """

        done = False

        conn = sqlite3.connect(inputFile)

        first_offset = offset
        new_offset = first_offset

        end_window = offset + window_size
        # The limit starts as the created in the constructor
        limit = self.size

        while not done:
            # If the number of rows left to be searched is smaller than the limit of the constructor
            if end_window - new_offset < self.size:
                # Set the limit to amount of rows left to search
                limit = end_window - new_offset

            # print("next batch:", new_offset.__str__(), "-", limit)

            conn = sqlite3.connect(inputFile)

            new_value = pd.read_sql_query("SELECT " + ','.join(columns) + " FROM price_hist LIMIT "
                                          + limit.__str__() + " OFFSET " + new_offset.__str__(), conn)

            if new_offset is first_offset:
                df = new_value
            else:
                df = df.append(new_value, ignore_index=True)

            if new_value.size < self.size:
                done = True

            new_offset += self.size
            if new_offset > end_window:
                break
        return df
