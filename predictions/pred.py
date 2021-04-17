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
plt.ylabel('Close price INR', fontsize = 15)
plt.plot(train['Close'], color = 'midnightblue')
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'upper left')
plt.show()
