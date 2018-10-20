# Multiple Linear Regression

# Assumptions of a Linear Regression:
# 1. Linearity
# 2. Homoscadasticity
# 3. Multivariate Normality
# 4. Independence of errors
# 5. Lack of multicollinearity

# Dummy Variables

# Does the state the company is located in matter?
# y = b0 + b1x1 + b2*x2 + b3x3 + b4D1

# Dummy variable Trap!!!
# D2 = 1 - D1, always omit one dummy variable

# How to build models step by step
# x1 => y (Simple Linear Regression)
# x1, x2, x3, x4, x5 => y

# 1. All-in (Use if you have prior knowledge, or you have to)
# 2. Backward Elmination ()
#   a. Step 1: Select a significance level to stay in the model (eg SL = 0.05)
#   b. Step 2: Fit the full model with all possible predictors
#   c. Step 3:Consider the predictor with the highest P-value. If p> SL go to Step 4
# 3. Forward Selection
#   a. Step 1: Select a sig level to enter the model SL = 0.05
#   b. Step 2: Fit all simple regression models slect the one with the lower p value
#   c. Step 3: Keep this variable and fit all possible models with one extra predictor added to the one you already have
#   d. Step 4: Consider the predictor with the lowest pvalue. If p<sl go to step 3, otherwise go to fin
# 4. Bidirectional Elimination (Combines the Backward Elimination and Forward Selection)
# 5. Score Comparision

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Data
# Endoing the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# Backward Elimination Time!!!

# X2 was above the significance level, had highest pvalue
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Best Indepedent Variables are R&D spent and Marketing Spent
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Import the library
import statsmodels.formula.api as sm
# Create Function, pass parameters x and sl
def backwardElimination(x, sl):
    
    numVars= len(x[0])
    for i in range(x, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

sl = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_opt, sl)