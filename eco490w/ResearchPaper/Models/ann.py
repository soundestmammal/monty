# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_V3-3.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset[dataset.INCOM_R != 9]
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
onehotencoder = OneHotEncoder()
y = y.reshape(-1,1)
y = onehotencoder.fit_transform(y).toarray()


#labelencoder_X_0 = LabelEncoder()
#X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
#
#
#labelencoder_X_1 = LabelEncoder()
#X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
#onehotencoder = OneHotEncoder(categorical_features=[4])
#X = onehotencoder.fit_transform(X).toarray()
#
#labelencoder_X_2 = LabelEncoder()
#X[:, 3] = labelencoder_X_2.fit_transform(X[:, 2])
#
#labelencoder_X_2 = LabelEncoder()
#X[:, 4] = labelencoder_X_2.fit_transform(X[:, 2])
#
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 7))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred * 1

y_test = y_test.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)