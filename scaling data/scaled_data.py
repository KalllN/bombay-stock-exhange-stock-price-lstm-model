#Importing the libraries
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#Getting the data
df = web.DataReader('^BSESN', data_source = 'yahoo', start = '2012-01-01', end = '2021-04-15')

#creating new datafile with only 'close' column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values

#Scaling data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)
