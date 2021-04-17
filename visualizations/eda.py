import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Getting the data
df = web.DataReader('^BSESN', data_source = 'yahoo', start = '2012-01-01', end = '2021-04-15')

#Visualize the closing price history
plt.figure(figsize = (16, 8))
plt.title('Closing price history')
plt.plot(df['Close'], color = 'forestgreen')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Close price INR', fontsize = 15)
plt.show()
