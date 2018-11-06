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


# Part 2 Building the RNN

# Part 3 Making the predictions and visualizing the results