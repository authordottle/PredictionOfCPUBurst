# /********* index.py ***********/

# Import library
import pandas as pd  # Data manipulation
import numpy as np  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Visualization

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from numpy import mean
from numpy import std
from numpy import absolute

# print('version: {}'.format(pd.__version__))

# Ask user for input
input = input(
    "Enter the platform data to start (enter either `linux` or `macBook`): ")

# Load the csv file
if (input == "macBook"):
    # pid,name,username,memory_percent,cpu_percent,nice,create_time,mem_rss,mem_vms,mem_pfaults,mem_pageins,num_ctx_switches_voluntary,num_ctx_switches_involuntary,utime,stime,cutime,cstime
    df = pd.read_csv('../data_collecting/darwin_processes.csv')
    X_drop_columns = ['pid', 'stime']
    y_column = 'stime'
    print("macBook")
elif (input == "linux"):
    # PID, NAME, ELAPSED_TIME, TOTAL_TIME, utime, stime, start_time, uptime
    df = pd.read_csv('../data_collecting/linux_log_file.csv')
    X_drop_columns = ['PID', 'stime']
    y_column = 'stime'
    print("linux")
else:
    print("Invalid input")

# Check for df
# print(df.columns)
# print(df.dtypes)

# Check one column vs another plot
# sns.lmplot(x='utime', y='stime', data=df, aspect=2, height=6)
# plt.xlabel('utime: as Independent variable')
# plt.ylabel('stime: as Dependent variable')
# plt.title('utime Vs stime')
# plt.show()

# Filter obj type columns
obj_type_cols = df[[
    i for i in df.columns if df[i].dtype == 'object']]
# print(obj_type_cols)

# Find the correlation among the columns
corr = df.corr()
# print(corr)

# Show correlation plot
# plt.figure(figsize=(12, 4))
# sns.heatmap(corr, cmap='Wistia', annot=True)
# plt.title('Correlation in the dataset')
# plt.show()

# Drop un-necessary information
#

# Check for missing value
# plt.figure(figsize=(12, 4))
# sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
# plt.title('Missing value in the dataset')
# plt.show()

# ML algorithms cannot work with categorical data directly, categorical data must be converted to number.
# 1. Label Encoding
# 2. One hot encoding
# 3. Dummy variable trap

# 1. Label Encoding (Future work)
le = LabelEncoder()  # Code categories into 0,1,2.....

# 2. One-Hot Encoding (Future work)
ohe = OneHotEncoder()

# 3. Dummy variable trap
df_encode = pd.get_dummies(data=df, prefix='OHE', prefix_sep='_',
                           columns=obj_type_cols.columns.tolist(),
                           drop_first=True,
                           dtype='int8')
# Verify the dummay variable process
# print('Columns in original data frame:', df.columns.values)
# print('Number of rows and columns in the dataset:', df.shape)
# print('Columns in data frame after encoding dummy variable: ',
#       df_encode.columns.values)
# print('Number of rows and columns in the dataset: ', df_encode.shape)

# Assign columns variables to X and y
X = df_encode.drop(X_drop_columns, axis=1)  # Independent variable
# utime: CPU time spent in user code
# stime: CPU time spent in kernel code
y = df_encode[y_column]  # Dependent variable

# Train test split
# test_size: 25% of the data will go to the test set, whereas the remaining 75% to the training set
# random_state: using this parameter makes sure that anyone who re-runs your code will get the exact same outputs.
# Popular integer random seeds are 0 and 42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# print(X_train)
# print(X_train.dtypes)
# print(X_train.shape)
# print(X_test)
# print(y_train)
# print(y_test)

######################################### Linear Regression ###############################################
lr = LinearRegression()
lr.fit(X_train, y_train)

# Try to build model for Normal Equation
# 1: Add x0 = 1 to train and test dataset based on their # of rows
X_train_0 = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_0 = np.c_[np.ones((X_test.shape[0], 1)), X_test]
# print(X_train_0)
# print(X_test_0)

# Check eligibility for normal equation
if (np.linalg.det(X_train_0.T.dot(X_train_0)) == 0):
    print("This step creates singular matrix. Thus, normal equation steps will be skipped.")
else:
    # 2: Build model using normal equation
    # or theta = np.matmul(np.linalg.inv(np.matmul(X_train_0.T, X_train_0)), np.matmul(X_train_0.T, y_train))
    train_theta = np.linalg.inv((X_train_0.T).dot(
        X_train_0)).dot(X_train_0.T).dot(y_train)
    # Compare 2 theta
    parameter_df = pd.DataFrame(
        {'parameter': ['theta_'+str(i) for i in range(X_train_0.shape[1])],
         'columns': ['intersect:x_0=1'] + list(X.columns.values),
         'train_theta': train_theta,
         'sklearn_theta': [lr.intercept_]+list(lr.coef_)
         })
    # print(parameter_df)

# Predict by sklearn linear regression model
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluation: MSE
# variables contain the performance metrics MSE and R2 for models build using linear regression on the training set
# MSE: the squared distance between actual and predicted values. Squared to avoid the cancellation of negative terms.
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# or lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_train_r2 = lr.score(X_train, y_train)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# or lr_test_r2 = r2_score(y_test, y_lr_test_pred)
lr_test_r2 = lr.score(X_test, y_test)

# Data visualization of prediction results for lr
# plt.figure(figsize=(5, 5))
# plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
# # A trend line to the plot
# z = np.polyfit(y_train, y_lr_train_pred, 1)
# # print(z)
# p = np.poly1d(z)
# plt.plot(y_train, p(y_train), "#F8766D")
# plt.title('Using linear regression', fontsize=15)
# plt.ylabel('Predicted')
# plt.xlabel('Experimental')
# plt.show()

######################################### Ridge Regression ###############################################
rr = Ridge(alpha=1.0)
rr.fit(X_train, y_train)
y_rr_train_pred = rr.predict(X_train)
y_rr_test_pred = rr.predict(X_test)

# Evaluation: MSE
# variables contain the performance metrics MSE and R2 for models build using linear regression on the training set
# MSE: the squared distance between actual and predicted values. Squared to avoid the cancellation of negative terms.
rr_train_mse = mean_squared_error(y_train, y_rr_train_pred)
# or rr_train_r2 = r2_score(y_train, y_rr_train_pred)
rr_train_r2 = rr.score(X_train, y_train)
rr_test_mse = mean_squared_error(y_test, y_rr_test_pred)
# or rr_test_r2 = r2_score(y_test, y_rr_test_pred)
rr_test_r2 = rr.score(X_test, y_test)

# Data visualization of prediction results for rr
# plt.figure(figsize=(5, 5))
# plt.scatter(x=y_train, y=y_rr_train_pred, c="#7CAE00", alpha=0.3)
# # A trend line to the plot
# z = np.polyfit(y_train, y_rr_train_pred, 1)
# # print(z)
# p = np.poly1d(z)
# plt.plot(y_train, p(y_train), "#F8766D")
# plt.title('Using ridge regression', fontsize=15)
# plt.ylabel('Predicted')
# plt.xlabel('Experimental')
# plt.show()

# # define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# scores = cross_val_score(lasso, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # force scores to be positive
# scores = absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

######################################### Random Forest ###############################################
# build a random forest with 1000 decision trees
rfr = RandomForestRegressor(n_estimators=1000, random_state=42)
rfr.fit(X_train, y_train)
y_rfr_train_pred = rfr.predict(X_train)
y_rfr_test_pred = rfr.predict(X_test)
# print(y_train)
# print(y_rfr_train_pred)
# print(y_rfr_train_pred)

# Evaluation: MSE
# variables contain the performance metrics MSE and R2 for models build using random forest on the training set
rfr_train_mse = mean_squared_error(y_train, y_rfr_train_pred)
# or rfr_train_r2 = r2_score(y_train, y_rfr_train_pred)
rfr_train_r2 = rfr.score(X_train, y_train)
rfr_test_mse = mean_squared_error(y_test, y_rfr_test_pred)
# or rfr_test_r2 = r2_score(y_test, y_rfr_test_pred)
rfr_test_r2 = rfr.score(X_test, y_test)

# Data visualization of prediction results for rfr
# plt.figure(figsize=(5, 5))
# plt.scatter(x=y_train, y=y_rfr_train_pred, c="#7CAE00", alpha=0.3)
# # A trend line to the plot
# z = np.polyfit(y_train, y_rfr_train_pred, 1)
# # print(z)
# p = np.poly1d(z)
# plt.plot(y_train, p(y_train), "#F8766D")
# plt.title('Using random forest regressor', fontsize=15)
# plt.ylabel('Predicted')
# plt.xlabel('Experimental')
# plt.show()

######################################### Conclusion of Evaluation ###############################################
data = [
    ['Linear regression', lr_train_mse, np.sqrt((lr_train_mse)),
     lr_train_r2, lr_test_mse, np.sqrt((lr_test_mse)), lr_test_r2, lr_test_r2*100],
    ['Ridge regression', rr_train_mse, np.sqrt((rr_train_mse)),
     rr_train_r2, rr_test_mse, np.sqrt((rr_test_mse)), rr_test_r2, rr_test_r2*100],
    ['Random forest', rfr_train_mse, np.sqrt((rfr_train_mse)),
     rfr_train_r2, rfr_test_mse, np.sqrt((rfr_test_mse)), rfr_test_r2, rfr_test_r2*100],
]
results = pd.DataFrame(data,
                       columns=['Method', 'Training MSE', 'Training RMSE',
                                'Training R2', 'Test MSE', 'Test RMSE', 'Test R2', 'Accuracy %']
                       )
print(results)
