import pandas as pd
import numpy as np
from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Excel file
# ['model number', 'freq', 'L2', 'L3', 'memory', 'bench', 'single/multicore', 'score', 'perf']

# processData = pd.read_excel("data.xls")
# print(processData.columns)
# print(processData.dtypes)
# objTypeCols = processData[[i for i in processData.columns if processData[i].dtype == 'object']]; del(i)

# corrprocessData = processData.corr()
# print(corrprocessData)
# corrprocessData['PerformanceRating'].sort_values(ascending=False)
# corrprocessData['PerformanceRating'].sort_values(ascending=False).index[:-4:-1]
# # EmpEnvironmentSatisfaction, EmpLastSalaryHikePercent is having high Corr with PerformanceRating
# processData[corrprocessData['PerformanceRating'].sort_values(ascending=False).index[0:4]].head()
# processData[corrprocessData['PerformanceRating'].sort_values(ascending=False).index[[0,-1,-2,-3]]].head()

# # Group by Department and find the which Department has the Highest Performance rating
# processData.groupby("EmpDepartment")['PerformanceRating'].mean()

# # Check for Unique Departments in the Dataset
# processData['EmpDepartment'].value_counts()

# # Find the Number of Employees who have rating of 2,3,4
# processData["PerformanceRating"].value_counts()

# # Converting data type to 'category' - encoding

# # Label Encoding
# # from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# # le = LabelEncoder() # LabelEncoder - Code categories into 0,1,2.....
# # for i in objTypeCols.columns.values.tolist()[1:]:
# #     processData[i+"_coded"] = processData[i].astype('category')
# #     processData[i+"_coded"] = le.fit_transform(processData[i+"_coded"])

# # One Hot Encoding
# for i in objTypeCols.columns.values.tolist()[1:]:
#     dummyCols = pd.get_dummies(processData[i], prefix=i) # Convert 1/0 based on presence; C no. of columns
#     processData = processData.join(dummyCols)
# del dummyCols,i
# # ohe = OneHotEncoder()
# # tmp = OneHotEncoder(categorical_features=le.fit_transform(processData['MaritalStatus']))

# # Model building
# train, test = train_test_split(processData, test_size=0.3, random_state=123, stratify=processData["PerformanceRating"])

######################################## Random Forest ############################################
# read dataset
dataset = pd.read_csv('data_preparing/data/machine.csv')
x = dataset.iloc[:, 2:-2].values
y = dataset.iloc[:, -2].values

# split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Random Forest
rfr = RandomForestRegressor(n_estimators=50, max_features=None, random_state=0)
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)
r2_score = rfr.score(x_test, y_test)
print("Accuracy1:", r2_score*100, '%')
print("RMSE1: ", sqrt(mean_squared_error(y_test, y_pred)))

# predicting value
new_prediction = rfr.predict(np.array([[1100, 768, 2000, 0, 1, 1]]))
print("Prediction performance:", float(new_prediction))
new_prediction = rfr.predict(np.array([[143, 512, 5000, 0, 7, 32]]))
print("Prediction performance:", float(new_prediction))
new_prediction = rfr.predict((np.array([[64, 5240, 20970, 30, 12, 24]])))
print("Prediction performance:", float(new_prediction))
new_prediction = rfr.predict((np.array([[700, 256, 2000, 0, 1, 1]])))
print("Prediction performance:", float(new_prediction))

'''
OUTPUT:
Accuracy1: 82.3363526342748 %
RMSE1:  48.398874036330604   
Prediction performance: 14.92
Prediction performance: 38.92
Prediction performance: 190.83333333333337
Prediction performance: 22.04666666666667 
'''

######################################## Linear Regression ############################################
# read dataset
dataset = pd.read_csv('data_preparing/data/machine.csv')
x = dataset.iloc[:, 2:-2].values
y = dataset.iloc[:, -2].values

# split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

# Linear Regression
lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
r2_score = lr.score(x_test, y_test)
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

'''
OUTPUT:
Accuracy1: 73.34625718871324 %
RMSE1:  59.453053748648685
Prediction performance: 26.735479830340026
Prediction performance: 45.48104224752923
Prediction performance: 186.95419502672712
Prediction performance: -4.773525537188831
'''