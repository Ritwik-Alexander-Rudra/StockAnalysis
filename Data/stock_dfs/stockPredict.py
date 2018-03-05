import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import datetime
import pickle
import csv
import calendar
import time
import arrow
import requests
import bs4 as bs
quandl.ApiConfig.api_key = '5bJJjfrSVFqoMTeNsuCJ'
tickers = []
def makeTickerArray():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")

    table = soup.find('table', {'class': 'wikitable sortable'})
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

def makeGraph(company_ticker):
    style.use('ggplot')
    df = pd.read_csv(company_ticker+".csv")
    dates = df['Date']
    df = df[[ 'Open', 'High',  'Low',  'Adj Close', 'Volume']]
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
    forecast_col = 'Adj Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.1 * len(df)))

    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out:]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    #COMMENTED OUT:
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    with open('linearregression.pickle','wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = arrow.get(last_date).timestamp
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    df['Adj Close'].plot()
    df['Forecast'].plot()
    x = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    y = range(len(x))
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("Historical and Projected Stock Price of " + company_ticker)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.plot(x)
    plt.gcf().autofmt_xdate()
    plt.show()


makeTickerArray()
for i in range(10):
    makeGraph(tickers[i])
