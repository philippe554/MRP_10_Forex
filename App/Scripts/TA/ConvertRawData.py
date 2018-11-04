"""
    Converts the downloaded raw data into the format used by TA
    [Timestamp,Open,High,Low,Close,Volume_USD,Volume_Currency,Weighted_Price]
"""

import csv
import pandas as pd
from ta import *
import datetime

inputFile = '../../Data/2018oct.csv'
outputFile = '../../Data/raw_data_ta.csv'

# Load data
df = pd.read_csv(inputFile, sep=';')


with open(outputFile, mode='w', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['Timestamp','Open','High','Low','Close','Volume_USD','Volume_Currency','Weighted_Price'])
    for index, row in df.iterrows():
        d = datetime.datetime.strptime(row.DateTimeStamp, '%Y%m%d %H%M%S')
        employee_writer.writerow([
            d.timestamp().__int__(),
            row.BarOPENBid,                         # 'Open'
            row.BarHIGHBid,                         # 'High'
            row.BarLOWBid,                          # 'Low'
            row.BarCLOSEBid,                        # 'Close'
            25,                             # 'Volume_USD'
            150,                             # 'Volume_Currency'
            # row.Volume,                             # 'Volume_USD'
            # row.Volume,                             # 'Volume_Currency'
            (row.BarOPENBid+row.BarCLOSEBid)/2])    # 'Weighted_Price'
employee_file.close()
