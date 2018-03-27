import pandas as pd
import quandl,math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = 'mxCZaX7q5PPoFWsSjbsS'
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HighLowPercent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PercentChange'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HighLowPercent', 'PercentChange','Adj. Volume',]]

predictCol = 'Adj. Close'
df.fillna(-99999, inplace=True)

predictOut = int(math.ceil(0.01*len(df)))

df['label'] = df[predictCol].shift(-predictOut)

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-predictOut:]
x = x[:-predictOut:]
df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y, test_size=0.2)

classifier = LinearRegression(n_jobs=-1)
# classifier = svm.SVR(kernal = 'poly')
classifier.fit(x_train,y_train)
accuracy = classifier.score(x_test,y_test)

predictSet = classifier.predict(x_lately)

df['predict'] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in predictSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneDay
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print('Accuracy : ',accuracy)
df['Adj. Close'].plot()
df['predict'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()