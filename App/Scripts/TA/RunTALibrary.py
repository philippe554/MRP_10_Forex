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
open = "Open"
high = "High"
low = "Low"
close = "Close"
fillna = True
# df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_USD", fillna=True)
# df = add_volume_ta(df, high, low, close, volume, fillna=fillna)
# df = add_momentum_ta(df, high, low, close, volume, fillna=fillna)
df = add_volatility_ta(df, high, low, close, fillna=fillna)
df = add_trend_ta(df, high, low, close, fillna=fillna)
df = add_others_ta(df, close, fillna=fillna)

# Momentum TA
df['momentum_rsi'] = rsi(df[close], n=14, fillna=fillna)
# df['momentum_mfi'] = money_flow_index(df[high], df[low], df[close], df[volume], n=14, fillna=fillna)
df['momentum_tsi'] = tsi(df[close], r=25, s=13, fillna=fillna)
df['momentum_uo'] = uo(df[high], df[low], df[close], fillna=fillna)
df['momentum_stoch'] = stoch(df[high], df[low], df[close], fillna=fillna)
df['momentum_stoch_signal'] = stoch_signal(df[high], df[low], df[close], fillna=fillna)
df['momentum_wr'] = wr(df[high], df[low], df[close], fillna=fillna)
df['momentum_ao'] = ao(df[high], df[low], fillna=fillna)

df.to_csv('2018oct_testing_novolume.csv')

print(df.columns)
print(len(df.columns))