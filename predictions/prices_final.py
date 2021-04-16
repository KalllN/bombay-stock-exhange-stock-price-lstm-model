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

jpm1 = web.DataReader('JPM', data_source = 'yahoo', start = '2020-12-18', end = '2020-12-18')
print(jpm1['Close'])
print(pred_price)
