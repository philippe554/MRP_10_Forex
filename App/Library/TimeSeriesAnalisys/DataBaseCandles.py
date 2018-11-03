import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Create canvas from csv
df = pd.read_csv("2018oct.csv", delimiter=";")
# Column names
# DateTimeStamp 
# BarOPENBid
# BarHIGHBid
# BarLOWBid
# BarCLOSEBid
# Volume
# If the csv file does not have this values in the first row then the file should be modified and introduced:
# DateTimeStamp;BarOPENBid;BarHIGHBid;BarLOWBid;BarCLOSEBid;Volume

df.index = pd.to_datetime(df.DateTimeStamp,format='%Y%m%d %H%M%S',errors='coerce') 
# print(df.head())
price = df[["BarOPENBid", "BarHIGHBid", "BarLOWBid", "BarCLOSEBid"]]
train = price[:len(price) - 1000]
test = price[len(price) - 1000:]
# print(price.head())
# print(test.values[1:10])

# print(np.corrcoef(train.values[0: 0 + len(test)], test.values)

corr = []
for window in range(len(train) - len(test)):
	rowCorr = []
	for column in range(4):
		# print()
		# print('train')
		trainValues = train.values[window : window + len(test), column]
		# print(trainValues)
		# print("FULL Window: " +str(train.values[:, window : window + len(test)]))
		# print('test')
		testValues = test.values[:, column]
		# print(testValues)
		rowCorr.append(np.corrcoef(trainValues, testValues)[0,1])
		# print("WINDOW: "+str(window))

	corr.append(np.mean(rowCorr))		

max = np.argmax(corr)
min = np.argmin(corr)

plt.plot(train['BarOPENBid'][max:max+len(test)], label = 'Maximum correlation')
plt.plot(train['BarOPENBid'][min:min+len(test)], label = 'Minimum correlation')
plt.plot(test['BarOPENBid'], label= 'Window')
plt.legend(loc='best')
plt.figure()
plt.plot(range(len(test)), train['BarOPENBid'][max:max+len(test)], label = 'Maximum correlation')
plt.plot(range(len(test)), train['BarOPENBid'][min:min+len(test)], label = 'Minimum correlation')
plt.plot(range(len(test)), test['BarOPENBid'], label= 'Window')
plt.legend(loc='best')
plt.show()
