# /********* index.py ***********/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Load the Excel file
dataset = pd.read_csv('../data_collecting/darwin_processes.csv')

# Assigning couple columns variables to X
X = dataset.iloc[:, 5:-2].values
# Assigning the last 2 column variable to y
y = dataset.iloc[:, -2].values
print(dataset.columns)
print(dataset.dtypes)
# print(x)
# print(y)
objTypeCols = dataset[[
    i for i in dataset.columns if dataset[i].dtype == 'object']]

#
corrprocessData = dataset.corr()
# print(corrprocessData)

# Dropping un-necessary information

# Model building
# test_size: 25% of the data will go to the test set, whereas the remaining 75% to the training set
# random_state: using this parameter makes sure that anyone who re-runs your code will get the exact same outputs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

######################################### Linear Regression ###############################################
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
r2_score = lr.score(X_test, y_test)
print("Accuracy1:", r2_score*100, '%')
print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# predicting value
new_prediction = lr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))
new_prediction = lr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
print("Prediction performance:", float(new_prediction))
new_prediction = lr.predict(np.array([[64, 5240, 20970, 30, 12, 24]]))
print("Prediction performance:", float(new_prediction))
new_prediction = lr.predict(np.array([[700, 256, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))

######################################### Random Forest ###############################################
Rr = RandomForestRegressor(n_estimators=50, max_features=None, random_state=0)

Rr.fit(X_train, y_train)

y_pred = Rr.predict(X_test)
r2_score = Rr.score(X_test, y_test)
print("Accuracy1:", r2_score*100, '%')
print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# predicting value
new_prediction = Rr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))
new_prediction = Rr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
print("Prediction performance:", float(new_prediction))
new_prediction = Rr.predict((np.array([[64, 5240, 20970, 30, 12, 24]])))
print("Prediction performance:", float(new_prediction))
new_prediction = Rr.predict((np.array([[700, 256, 2000, 0, 1, 1]])))
print("Prediction performance:", float(new_prediction))
