import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
#^wants dates as well not just ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')


df = pd.read_csv('tsla.csv', parse_dates = True, index_col = 0)
#Make new columns
#Moving average is 100 past day prices and takes an average
#.rolling preps to start looking at past data
#window gives time frame^
#rolling looks behind
#df['100ma'] = df['Open'].rolling(window=100, min_periods = 0).mean()
#df.dropna(inplace=True)
#dropna eliminates invalid datapoints
#inplace modifies inplace; you don't have to reassign
#print (df.tail())
#each matplotlib has figures
#figures have multiple subplots
#subplots are axis
#look at subplots tutorial later

df_ohlc = df['Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
#print(df_ohlc.head())

#You want true volume not average volume over 10 days
#Can be resampled for however min
#ohlc = open high low close

print(df_ohlc.head())



ax1 = plt.subplot2grid((6, 1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6, 1), (5,0), rowspan = 1, colspan = 1, sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width = 2, colorup = 'g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0) #graphs y from 0 to volume values
#always got to map to make readable
#^Makes dates look beautiful
#can be modified for color etc (look at documentation)
#ax1.plot(df.index, df['Open'])
#ax2.bar(df.index, df['Volume'])
plt.show()
