"""
    Compute all features
"""
from ta import *
import datetime
import pandas as pd
import sqlite3

inputFile = '../../Data/price_hist.db'
targetFile = 'price_hist_ta.db'



done = False
# offset = 0 #next batch: 560000
offset = 560000
size = 30000
while not done:
    print("next batch: " + offset.__str__())
    conn = sqlite3.connect(inputFile)
    # df = pd.read_sql_query("SELECT * FROM price_hist LIMIT 10 OFFSET 6000000", conn)
    df = pd.read_sql_query("SELECT * FROM price_hist LIMIT "+size.__str__()+" OFFSET "+ offset.__str__(), conn)
    offset += size

    if df.shape != (0,7):
        # Add all ta features filling nans values
        open = "BarOPEMBid"
        high = "BarHIGHBid"
        low = "BarLOWBid"
        close = "BarCLOSEBid"
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

        # append to target db
        conn.close()
        connT = sqlite3.connect(targetFile)
        df.to_sql('price_hist', connT, if_exists='append')
        connT.close()

    else:
        done = True

print("done. offset = " + offset.__str__())