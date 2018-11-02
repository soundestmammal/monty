# Artificial NN Titanic Problem

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
example = pd.read_csv('gender_submission.csv')

X_train = train_dataset.iloc[:, [2,4,5,9]].values
y_train = train_dataset.iloc[:, 1].values
X_test = test_dataset.iloc[:, [1,3,4,8]].values
y_test = test_dataset.iloc[:, 1].values

# train_dataset.isnull().sum()
# test_dataset.isnull().sum()

from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values = 'NaN', strategy="median", axis=0)
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_train = LabelEncoder()
X_train[:, 1] = labelencoder_train.fit_transform(X_train[:, 1])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_train_2 = LabelEncoder()
X_train[:, 0] = labelencoder_train_2.fit_transform(X_train[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

from sklearn.preprocessing import Imputer
imputer_test = Imputer(missing_values = 'NaN', strategy="median", axis=0)
imputer_test = imputer_test.fit(X_test[:, 1:2])
X_test[:, 1:2] = imputer_test.transform(X_test[:, 1:2])

from sklearn.preprocessing import Imputer
imputer_test_fare = Imputer(missing_values = 'NaN', strategy="median", axis=0)
imputer_test_fare = imputer_test_fare.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer_test_fare.transform(X_test[:, 2:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_test = LabelEncoder()
X_test[:, 0] = labelencoder_test.fit_transform(X_test[:, 0])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))

# Add the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Add epochs
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

