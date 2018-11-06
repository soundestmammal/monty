# This is a Recurrent Neural Network

# Part 1 is to preprocess the data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv');
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the Data
# Adding another layer is adding another dimension
# in the example we can have a tensor or whatever we want
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 Building the RNN

# Part 3 Making the predictions and visualizing the results