# /********* index.py ***********/

import pandas as pd
import seaborn as sns
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
dataset = pd.read_csv('../data_collecting/darwin_processes.csv')

# Assigning couple columns variables to X
X = dataset.iloc[:, 3:-4]
# Assigning the last 2 column variable to y
y = dataset.iloc[:, -4:]
# print(dataset.columns)
# print(dataset.dtypes)
print(X)
print(y)

objTypeCols = dataset[[
    i for i in dataset.columns if dataset[i].dtype == 'object']]

#
corrprocessData = dataset.corr()
print(corrprocessData)

# Dropping un-necessary information

# Model building
# test_size: 25% of the data will go to the test set, whereas the remaining 75% to the training set
# random_state: using this parameter makes sure that anyone who re-runs your code will get the exact same outputs. Popular integer random seeds are 0 and 42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

######################################### Linear Regression ###############################################
lr = LinearRegression()

lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Model performance
# variables contain the performance metrics MSE and R2 for models build using linear regression on the training set
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse,
                          lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE',
                      'Training R2', 'Test MSE', 'Test R2']
print(lr_results)

# Data visualization of prediction results
# plt.figure(figsize=(5, 5))
# plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
# z = np.polyfit(y_train, y_lr_train_pred, 1)
# p = np.poly1d(z)
# plt.plot(y_train, p(y_train), "#F8766D")
# plt.ylabel('Predicted')
# plt.xlabel('Experimental')


# y_pred = lr.predict(X_test)
# r2_score = lr.score(X_test, y_test)

# print("Accuracy1:", r2_score*100, '%')
# print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# # predicting value
# new_prediction = lr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
# print("Prediction performance:", float(new_prediction))
# new_prediction = lr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
# print("Prediction performance:", float(new_prediction))
# new_prediction = lr.predict(np.array([[64, 5240, 20970, 30, 12, 24]]))
# print("Prediction performance:", float(new_prediction))
# new_prediction = lr.predict(np.array([[700, 256, 2000, 0, 1, 1]]))
# print("Prediction performance:", float(new_prediction))

# ######################################### Random Forest ###############################################
# Rr = RandomForestRegressor(n_estimators=50, max_features=None, random_state=0)

# Rr.fit(X_train, y_train)

# y_pred = Rr.predict(X_test)
# r2_score = Rr.score(X_test, y_test)
# print("Accuracy1:", r2_score*100, '%')
# print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# # predicting value
# new_prediction = Rr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
# print("Prediction performance:", float(new_prediction))
# new_prediction = Rr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
# print("Prediction performance:", float(new_prediction))
# new_prediction = Rr.predict((np.array([[64, 5240, 20970, 30, 12, 24]])))
# print("Prediction performance:", float(new_prediction))
# new_prediction = Rr.predict((np.array([[700, 256, 2000, 0, 1, 1]])))
# print("Prediction performance:", float(new_prediction))
