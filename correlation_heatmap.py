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

import pandas_datareader.data as web
import datetime as dt

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    #BS object^
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers
#save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(1990,1,1)
    end = dt.datetime(2018,3,1)

    for ticker in tickers:
        print (ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'google', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
                print(ticker)
            except:
                print("Try again")
                continue
        else:
            print('Already have {}'.format(ticker))

#get_data_from_yahoo()

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    #Empty dataframe
    main_df = pd.DataFrame()

    #Gets all data for each ticker
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace = True)

        df.rename(columns = {'Low': ticker}, inplace = True)
        #Removes unnecessary columns
        df.drop(['Close', 'Open', 'High', 'Volume'], 1, inplace = True)

        if main_df.empty:
            main_df = df
        else:
            #Handles missing data points
            main_df = main_df.join(df, how = 'outer')

        if count % 10 == 0:
            print(count)

        #print(main_df.head())
        main_df.to_csv('sp500_join_low.csv')
#compile_data()

def visualize_data():
    style.use('ggplot')
    df = pd.read_csv('sp500_join_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    #df_corr = df.corr()
    df.set_index('Date', inplace = True)
    df_corr = df.pct_change().corr()
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor = False)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    #If you want to map out covariance comment out below
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
visualize_data()
