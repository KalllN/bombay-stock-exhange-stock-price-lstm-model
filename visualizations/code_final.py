#Stock Price Prediction using LSTM
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
df = web.DataReader('JPM', data_source = 'yahoo', start = '2012-01-01', end = '2020-12-31')
df

#Visualize the closing price history
plt.figure(figsize = (16, 8))
plt.title('Closing price history')
plt.plot(df['Close'], color = 'forestgreen')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Close price USD($)', fontsize = 15)
plt.show()

#creating new datafile with only 'close' column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values

#80% training data, rounded up using math.ceil
training_model_len = math.ceil(len(dataset) * 0.8)

#Scaling data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

#Creating training dataset
train_data = scaled_data[0: training_model_len, :]

#Splitting data into x_train & y_train
x_train, y_train = [], []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i <= 60:
    print(x_train)
    print(y_train)
    print()

#Convert x_train & y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#Creating testing dataset
test_data = scaled_data[training_model_len - 60:, :]
#Creating the datasets x_test and y_test
x_test, y_test = [], dataset[training_model_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#converting data to numpy array
x_test = np.array(x_test)

#Reshaping the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#getting the models predicted price values
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

#getting RMSE
rmse = np.sqrt(np.mean(pred - y_test)**2)
rmse

#plotting data
train = data[:training_model_len].copy()
valid = data[training_model_len:].copy()
valid['Predictions'] = pred.copy()

plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Close price USD($)', fontsize = 15)
plt.plot(train['Close'], color = 'midnightblue')
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'upper left')
plt.show()

valid

#Get the quote
jpm = web.DataReader('JPM', data_source = 'yahoo', start = '2012-01-01', end = '2020-12-17')

new_df = jpm.filter(['Close'])

#getting last 60 day closing price value
last_60_days = new_df[-60:].values

#scaling the data
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#getting predicted scale price
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

jpm1 = web.DataReader('JPM', data_source = 'yahoo', start = '2020-12-18', end = '2020-12-18')
print(jpm1['Close'])
