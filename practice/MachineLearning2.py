import bs4 as bs
import numpy as np
import pickle
#Saves objects^
import requests
import matplotlib.pyplot as plt
from matplotlib import style
import os
#Creates directories^
import pandas as pd
from sklearn import preprocessing, cross_validation, svm #cross_validation shuffles
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
import datetime
import math
import time

df = pd.read_csv('tsla.csv')

df = df[['Open','High','Low','Close','Volume']]
#HL_PCT is high low percent change
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
#open close percent change
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
print(df.head())

forecast_col = 'Close'
df.fillna('-99999', inplace = True)

#How many days you want to project out
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

#Drop removes and returns
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
