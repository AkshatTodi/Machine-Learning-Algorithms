# Polynomial Regression

# Formula
# y = b0 + b1(X1**1) + b2(X1**2) + ... + bn(X1**n)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# When we build a machine learning model and specially regression model thgen we have to make our matix of feature to be considered all the 
#  time as a matrix means X should be in the form of (i,j) not as a vector (i,).
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)


# We are not spliting the dataset into test and train sets because our dataset is very small and in this condition we have
#  to make the obtain the maximun accuracy with this dataset.


# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X) # giving polynomial feature to X. 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
print(X_poly)
print(lin_reg_2.fit(X_poly, y))

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) # np.arange(lower bound, upper bound, incrementation)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression) and smooth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
salary_linear = lin_reg.predict([[6.5]])
print(salary_linear)

# Predicting a new result with Polynomial Regression
salary_polynomial_salary = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(salary_polynomial_salary)