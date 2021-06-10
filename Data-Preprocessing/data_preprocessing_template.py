# Data Preprocessing Template

# Importing the libraries
import numpy as np # It is a library which contains the methamatical tools
import matplotlib.pyplot as plt
import pandas as pd # It is used to import datasets and manage the datasets


# Importing the dataset

dataset = pd.read_csv('Data.csv')
# When we build a machine learning model and specially regression model thgen we have to make our matix of feature to be considered all the 
#  time as a matrix means X should be in the form of (i,j) not as a vector (i,).
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)


# Taking care of missing data

#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # "mean" is tge default value of strategy parameter, axis = 0 is for column and axis = 1 is for row.
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # "mean" is tge default value of strategy parameter, there is no axis parameter in 'SimputerImputer' library.
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(imputer)
print(X)


# Encoding categorical data

# 'LabelEncoder' is used to encode the string valuedes into number
# 'OneHotEncoder' is used to create the domey variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#LabelEncoder_X = LabelEncoder()
#X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
#X = onehotencoder.fit_transform(X).toarray()
X = ct.fit_transform(X)

LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

print(X)
print(y)
print("\n")


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
print("\n")


# Fearture Scaling
''' For most machine learning models we don't need to apply feature scaling because their libraries or other classes include feature scaling in
    their algorithms but for some machine learning models like SVR, we have to apply it saperately because they don't have this property.'''
# Scaling those domey variables is not necessery(It depends on the context) but here we are appling the feature scaling on whole X matrix.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_train)
print(X_test)