"""
    Compute all features
"""
import pandas as pd
from ta import *

inputFile = '../../Data/raw_data_ta.csv'

# Load data
df = pd.read_csv(inputFile, sep=',')

# Clean nan values
df = utils.dropna(df)

print(df.columns)

# Add all ta features filling nans values
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_USD",
                                fillna=True)

df.to_csv('2018oct_testing.csv')

print(df.columns)
print(len(df.columns))