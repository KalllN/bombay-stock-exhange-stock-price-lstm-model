bse = web.DataReader('^BSESN', data_source = 'yahoo', start = '2012-01-01', end = '2021-04-15')

new_df = bse.filter(['Close'])

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

bse1 = web.DataReader('^BSESN', data_source = 'yahoo', start = '2021-04-16', end = '2021-04-16')
print(bse1['Close'])
print(pred_price)
