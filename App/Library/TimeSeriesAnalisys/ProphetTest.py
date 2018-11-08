import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

# test if QT is working
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

# Create canvas from csv
df = pd.read_csv("2018oct.csv", delimiter=";")
df = df.rename(columns={'DateTimeStamp': 'ds', 'BarCLOSEBid': 'y'})

train = df[:len(df) - 1000]
test = df[len(df) - 1000:]
print(train.head())
print(len(train))

m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=1000)
forecast = m.predict(future)

fig1 = m.plot(forecast)
