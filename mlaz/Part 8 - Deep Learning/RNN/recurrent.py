# This is a Recurrent Neural Network

# Part 1 is to preprocess the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv');
training_set = dataset_train.iloc[:, 1:2].values



# Part 2 Building the RNN

# Part 3 Making the predictions and visualizing the results