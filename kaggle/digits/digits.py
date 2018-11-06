import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# Import the Datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# y_train is the actual digits in a vector
y_train = train['label']
# X_train is the pixel values without the revealed Digit
X_train = train.drop(labels = ['label'], axis = 1)

# Because the pixels are between 0 and 255, we will normalize them to
# reduce the computation requirements needed during training
X_train = X_train/255
test = test/255

# Reshape image
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Since y train is a categotical variable
# they should not have ordinal significance
# We need to make sure that they are just classes

y_train = to_categorical(y_train, num_classes = 10)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)


# Part 1 - Building the CNN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = (2, 2), padding = 'Same', input_shape = (28, 28, 1), activation = 'relu'))



# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the digits
classifier.fit(X_train, y_train, batch_size=86, epochs=1)


