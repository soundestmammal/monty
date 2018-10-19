# Notes on Random Forest

# Step 1: Pick as random K data points from the Training Set.

# Step 2: Build the Decision Tree associated to these K data points

# Step 3: Choose the number Ntree of treees you want to build and repeat STEPS 1&2

'''Step 4: For a new data point, make each one Ntree trees predict 
the value of Y for the data point in question and assign the 
new data point the average across all of the predicted Y values.'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()