import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#This is the start and stop date for stock prices
#start = dt.datetime(2000,1,1)
#end = dt.datetime(2018,3,1)
#df is a dataframe
#.head prints out the earliest
#.tail prints out the latest
#df = web.DataReader('TSLA', 'google', start, end)
#print (df.tail())
#df.to_csv('tsla.csv')
#Date is an index in a df

df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)
#parse_dates allows you to make dates->indexes

#print(df.head())

#df.plot()
print(df[['Open', 'High']].head)
#Can put in desired columns
plt.show()
