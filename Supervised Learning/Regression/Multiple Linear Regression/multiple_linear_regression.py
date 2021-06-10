# Multiple Linear Regression

# Formula
# y = b0 + b1*X1 + b2*X2 + ... + bn*Xn 
# y -> dependent variable
# (X1, X2, ..., Xn) -> inedependent variable
# b0 -> Constant
# (b1, b2, ..., bn) -> Coefficients

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# When we build a machine learning model and specially regression model thgen we have to make our matix of feature to be considered all the 
#  time as a matrix means X should be in the form of (i,j) not as a vector (i,).
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Encoding categorical data

'''# 'LabelEncoder' is used to encode the string valuedes into number
# 'OneHotEncoder' is used to create the domey variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder_X = LabelEncoder()
X[:, 3] = LabelEncoder_X.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = 'passthrough')
#X = onehotencoder.fit_transform(X).toarray()
X = ct.fit_transform(X)

print(X)'''


# Avoiding the Dummy Variable Trap
X = X[:, 1:]
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
slr = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_test)
print(y_pred)
print("\n")
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# model accuracy
accuracy = slr.score(X_test, y_test)
print(accuracy)
print("\n\n")

# Building the optical model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # adding '1' to the all data lines at position '0'.
print(X)
# in matrix X_opt we mention all index saperately so the during the process of backward elimination the column which is not play the optimal role in the output y can be easily eliminated.
X_opt = X[:, [0,1,2,3,4,5]] # this optimal matrix here will contain only independent cariables that have high impact on the profits.
# step: 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step: 3
regressor_OLS.summary()

# step: 4
X_opt = X[:, [0,1,3,4,5]] # this optimal matrix here will contain only independent cariables that have high impact on the profits.
# step: 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step: 3
regressor_OLS.summary()

# step: 4
X_opt = X[:, [0,3,4,5]] # this optimal matrix here will contain only independent cariables that have high impact on the profits.
# step: 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step: 3
regressor_OLS.summary()

# step: 4
X_opt = X[:, [0,3,5]] # this optimal matrix here will contain only independent cariables that have high impact on the profits.
# step: 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step: 3
regressor_OLS.summary()

# step: 4
X_opt = X[:, [0,3]] # this optimal matrix here will contain only independent cariables that have high impact on the profits.
# step: 5
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step: 3
regressor_OLS.summary()