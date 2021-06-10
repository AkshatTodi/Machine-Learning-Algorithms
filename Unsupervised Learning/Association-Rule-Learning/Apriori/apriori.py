# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# here "header" is used to add the index into dataset because actuly this dataset had no index but when we read the data here the first line of dataset is converted into index.
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# In Apriori algorithm there is no need to split the data into train and test sets, here we are taking whole dataset together.

transactions = []
print(len(dataset))
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results)
#print(*[i for i in results],sep="\n")