# Building a Recurrent Neural Network

# There will be three parts to this

# 1. Data Preprocessing

# No surprise,import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Opening stock price
training_set = dataset_train.iloc[:, 1:2].values

# implement the feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Create the data structure with 60 time steps

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

batch_size = X_train.shape[0]
time_steps = X_train.shape[1]
dimensions = 1

# Add more features with more dimensions
X_train = np.reshape(X_train, (batch_size, time_steps, dimensions))

# 2. Build the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Add first LSTM layer and add Dropout reg

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (time_steps, dimensions)))
regressor.add(Dropout(0.2))

# Add second LSTM layer and add Dropout reg
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Add third LSTM layer and add Dropout reg
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Add fourth LSTM layer and add Dropout reg
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Add the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# 3. Making the prediction and visualizing the results

