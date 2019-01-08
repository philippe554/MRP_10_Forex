import os
import sqlite3

import pandas as pd

inputFile = os.path.expanduser('~/Google Drive/Group 10_ Forecast the FOREX market/Data/price_hist_ta.db')

conn = sqlite3.connect(inputFile)
new_offset = 0
size = 300000
done = False

while not done:
    print("next batch: " + new_offset.__str__())
    new_value = pd.read_sql_query("SELECT * FROM price_hist LIMIT "
                                  + size.__str__() + " OFFSET " + new_offset.__str__(), conn)

    if new_offset == 0:
        df = new_value
    else:
        df = df.append(new_value, ignore_index=True)

    if new_value.size < size:
        done = True

    new_offset += size

df.to_csv("price_hist.csv")
