import pandas as pd
from App.Helpers.AccessTaDB import AccessDB
import matplotlib.pyplot as plt


def get_MACD(offset=0):
    db = AccessDB(offset)
    return db.get_column(["trend_macd_diff"])

print(get_MACD(5000000).tail())