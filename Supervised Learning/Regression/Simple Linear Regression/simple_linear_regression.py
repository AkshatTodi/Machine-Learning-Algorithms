# Simple Linear Regression

# Formula 
# y = b0 + b1*X1 
# y -> dependent variable
# X1 -> independent variable
# b0 -> Constent
# b1 -> Coefficient


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# When we build a machine learning model and specially regression model thgen we have to make our matix of feature to be considered all the 
#  time as a matrix means X should be in the form of (i,j) not as a vector (i,).
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
slr = regressor.fit(X_train, y_train) # fit(Training data, Target data)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# model accuracy
accuracy = slr.score(X_test, y_test)
print(accuracy)
print("\n\n")

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()