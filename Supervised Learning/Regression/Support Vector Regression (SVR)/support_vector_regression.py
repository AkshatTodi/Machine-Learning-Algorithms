# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1) # reshapeing the y because we are applying Feature scalling on y also and for this y should be in matrix form (1,j).
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y),1))
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
''' there are four kernels
   'linear'(Linear Kernal),
   'poly'(Polynomial Kernal),
   'sigmoid'(Sigmoid Kernal),
   'rbf'(Gaussion Kernal Radial Basis Function)'''
regressor = SVR(kernel = 'rbf') # our problem is non_linear that's why we can take from 'rbf' or 'poly'
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)
# if we want to check the prediction of any perticular value then we have to transform it, and the argument of transform must be an array.
# Do inverse_transform to get original scale of the salary
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)


# Visualising the SVR results 
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the SVR results with inverse_transform
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff with inverse transform')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff and smooth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff with inverse transform and smooth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()