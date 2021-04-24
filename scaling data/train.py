#80% training data, rounded up using math.ceil
training_model_len = math.ceil(len(dataset) * 0.8)

#Creating training dataset
train_data = scaled_data[0: training_model_len, :]

#Splitting data into x_train & y_train
x_train, y_train = [], []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

#Convert x_train & y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
