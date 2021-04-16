model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#using rmse
