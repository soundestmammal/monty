# Logistic Regression Titanic Problem

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
example = pd.read_csv('gender_submission.csv')

X_train = train_dataset.iloc[:, [4,5]].values
y_train = train_dataset.iloc[:, 1].values
X_test = test_dataset.iloc[:, [3,4]].values
y_test = test_dataset.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values = 'NaN', strategy="mean", axis=0)
imputer_train = imputer_train.fit(X_train[:, 1:2])
X_train[:, 1:2] = imputer_train.transform(X_train[:, 1:2])

from sklearn.preprocessing import LabelEncoder
labelencoder_train = LabelEncoder()
X_train[:, 0] = labelencoder_train.fit_transform(X_train[:, 0])

from sklearn.preprocessing import Imputer
imputer_test = Imputer(missing_values = 'NaN', strategy="mean", axis=0)
imputer_test = imputer_test.fit(X_test[:, 1:2])
X_test[:, 1:2] = imputer_test.transform(X_test[:, 1:2])

from sklearn.preprocessing import LabelEncoder
labelencoder_test = LabelEncoder()
X_test[:, 0] = labelencoder_test.fit_transform(X_test[:, 0])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

d = {'PassengerId': identify, 'Survived': y_pred}

submission = pd.DataFrame(data=d)
submission.to_csv('attempt1.csv', encoding='utf-8', index=False)
