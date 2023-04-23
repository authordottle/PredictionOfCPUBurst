# /********* index.py ***********/

""" Import library """
from scipy import stats
import scipy as sp
import pandas as pd  # Data manipulation
import numpy as np  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Visualization
import statsmodels.api as sm
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression():
    ######################################### Linear Regression ###############################################
    updated_lr_df = updated_df

    """ Initialize """
    global multicollinearity_drop_columns

    """ Remove all rows that have outliers in, at least, one column """
    # For each column, it first computes the Z-score of each value in the column, relative to the column mean and standard deviation.
    # It then takes the absolute Z-score because the direction does not matter, only if it is below the threshold.
    # "all(axis=1)" ensures that for each row, all column satisfy the constraint.
    # Finally, the result of this condition is used to index the dataframe.
    updated_lr_df = updated_lr_df[(
        np.abs(stats.zscore(updated_df)) < 3).all(axis=1)]

    def calculate_vif(data):
        vif_df = pd.DataFrame(columns=['Var', 'Vif'])
        x_var_names = data.columns
        for i in range(0, x_var_names.shape[0]):
            y = data[x_var_names[i]]
            x = data[x_var_names.drop([x_var_names[i]])]
            x = sm.add_constant(x)
            results = sm.OLS(y, x).fit()
            r_squared = results.rsquared
            # print(r_squared)
            if r_squared == 1:
                print(x_var_names[i] +
                      " r-squared is 1 which causes infinity.")
                multicollinearity_drop_columns.append(x_var_names[i])
            else:
                vif = round(1/(1-r_squared), 2)
                if vif > 5:
                    multicollinearity_drop_columns.append(x_var_names[i])
                vif_df.loc[i] = [x_var_names[i], vif]
        return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)

    """ V) Calculate VIF to verify multicollinearity for LR """
    if multicollinearity_drop_columns == []:
        vif_df = calculate_vif(updated_lr_df)
        updated_lr_df = updated_lr_df.drop(
            multicollinearity_drop_columns, axis=1)
    else:
        updated_lr_df = updated_lr_df.drop(
            multicollinearity_drop_columns, axis=1)
        multicollinearity_drop_columns = []
        vif_df = calculate_vif(updated_lr_df)
        updated_lr_df = updated_lr_df.drop(
            multicollinearity_drop_columns, axis=1)
    # print(multicollinearity_drop_columns)
    # print(vif_df)
    # print(updated_lr_df)

    """ Assign columns variables to X and y """
    X_df = updated_lr_df.drop(X_drop_columns, axis=1)  # Independent variable

    # utime: CPU time spent in user code
    # stime: CPU time spent in kernel code
    y_df = updated_lr_df[y_column]  # Dependent variable

    """"""
    X = X_df.values
    y = y_df.values

    """ Train test split """
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

    """"""
    sc = StandardScaler()
    X_train_transform = sc.fit_transform(X_train)
    X_test_transform = sc.fit_transform(X_test)

    def get_theta_by_gradient_descent(X, y, learning_rate, iterations):
        y_new = np.reshape(y, (len(y), 1))
        cost_lst = []
        X_0 = np.c_[np.ones((len(X), 1)), X]
        theta = np.random.randn(len(X[0])+1, 1)
        m = len(X)
        for i in range(iterations):
            gradients = 2/m * X_0.T.dot(X_0.dot(theta) - y_new)
            theta = theta - learning_rate * gradients
            y_pred = X_0.dot(theta)
            cost_value = 1/(2*len(y))*((y_pred - y)**2)
            # Calculate the loss for each training instance
            total = 0
            for i in range(len(y)):
                total += cost_value[i][0]
                # Calculate the cost function for each iteration
            cost_lst.append(total)
        plt.plot(np.arange(1, iterations), cost_lst[1:], color='red')
        plt.title('Cost function Graph')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')

        return theta

    """ Find the multivarite regression model for the data set  """
    # Parameters: X: independent variables matrix; y: dependent variables matrix; learning_rate: learning rate of Gradient Descent; iterations: the number of iterations
    # Return: the final theta vector and the plot of cost function
    mlr_theta = get_theta_by_gradient_descent(
        X_train_transform, y_train.tolist(), 0.03, 30000)

    """ I. Training a Linear Regression model which is ordinary least squares Linear Regression by scikit-learn implementation """
    lr = LinearRegression()
    lr.fit(X_train_transform, y_train)

    """ Check they match or not """
    flat_mlr_theta_list = [item for sublist in mlr_theta for item in sublist]
    sub_list = [lr.intercept_]
    lr_info_list = [*sub_list, *(lr.coef_)]
    # print(flat_mlr_theta_list)
    # print(lr_info_list)

    """ Predict by sklearn linear regression model """
    y_lr_train_pred = lr.predict(X_train_transform)
    y_lr_test_pred = lr.predict(X_test_transform)
    # print(y_lr_train_pred)
    # print(y_lr_test_pred)

    ######################################### Evaluation ###############################################
    """ MSE and R2 """
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = lr.score(X_train_transform, y_train)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = lr.score(X_test_transform, y_test)

    """ Compute the cross-validation scores """
    # n_splits, default=5; n_repeats, default=10
    cv = RepeatedKFold(random_state=42)
    lr_mse_scores = cross_val_score(
        lr, X, y, cv=cv, scoring='neg_mean_squared_error')
    lr_rmse_scores = cross_val_score(
        lr, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    lr_mse_mean_scores = abs(lr_mse_scores.mean())
    lr_std_scores = lr_mse_scores.std()
    lr_rmse_mean_scores = abs(lr_rmse_scores.mean())
    lr_rmse_std_scores = lr_rmse_scores.std()

    ######################################### Validation ###############################################
    """ I) Check for Linearity """
    f = plt.figure(figsize=(14, 5))
    ax1 = f.add_subplot(121)
    sns.scatterplot(x=y_test, y=y_lr_test_pred, ax=ax1, color='r')
    ax1.set_title('Check for Linearity:\n Actual Vs Predicted value')

    """ II) Check for Residual normality & mean """
    ax2 = f.add_subplot(122)
    sns.histplot(data=(y_test - y_lr_test_pred), ax=ax2, color='b')
    ax2.axvline((y_test - y_lr_test_pred).mean(),
                color='yellow', linestyle='--')
    ax2.set_title('Check for Residual normality & mean: \n Residual eror')

    """ III) Check for Multivariate Normality by Quantile-Quantile plot """
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    _, (_, _, r) = sp.stats.probplot(
        (y_test - y_lr_test_pred), fit=True, plot=ax[0])
    ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')

    """ IV) Check for Homoscedasticity """
    sns.scatterplot(y=(y_test - y_lr_test_pred),
                    x=y_lr_test_pred, ax=ax[1], color='r')
    ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted')

    ######################################### Visualization ###############################################
    plt.figure(figsize=(5, 5))
    plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
    # A trend line to the plot
    z = np.polyfit(y_train, y_lr_train_pred, 1)
    # print(z)
    p = np.poly1d(z)
    plt.plot(y_train, p(y_train), "#F8766D")
    plt.title('Using linear regression', fontsize=15)
    plt.ylabel('Predicted')
    plt.xlabel('Experimental')

    # plt.show()

    ######################################### Conclusion of Evaluation ###############################################
    data = [
        ['Linear regression',
         # lr_train_mse, np.sqrt((lr_train_mse)), lr_train_r2,
         lr_test_mse, np.sqrt((lr_test_mse)), lr_test_r2, lr_mse_mean_scores, lr_rmse_mean_scores, lr_std_scores],
    ]
    results = pd.DataFrame(data,
                           columns=['Method',
                                    # 'Training MSE', 'Training RMSE', 'Training R2',
                                    'Test MSE', 'Test RMSE', 'Test R2', 'Cross-Val MSE Mean', 'Cross-Val RMSE Mean', 'Cross-Val MSE Standard Deviation']
                           )
    print(results)


def random_forest_regression():
    ######################################### Random Forest ###############################################
    updated_rfr_df = updated_df

    """ Assign columns variables to X and y """
    X_df = updated_rfr_df.drop(X_drop_columns, axis=1)  # Independent variable

    # utime: CPU time spent in user code
    # stime: CPU time spent in kernel code
    y_df = updated_rfr_df[y_column]  # Dependent variable

    """"""
    X = X_df.values
    y = y_df.values

    """ Train test split """
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

    """ Build a random forest with some decision trees """
    # Initialize
    param_grid = {
        'n_estimators': [300, 500, 1000],
        'max_features': ['sqrt', 'log2', 20],
        'max_depth': [7],
        'random_state': [42, 18, 2017],
        'min_samples_leaf': [1]
    }
    # n_splits, default=5; n_repeats, default=10
    cv = RepeatedKFold(random_state=42)
    mse_scoring = "neg_mean_squared_error"
    rmse_scoring = 'neg_root_mean_squared_error'
    n_estimators = 500
    max_depth = 7
    max_features = "sqrt"
    random_state = 42
    min_samples_leaf = 1

    # Find best_params_ by Grid Search function
    cv_rfr = GridSearchCV(estimator=RandomForestRegressor(),
                          param_grid=param_grid, cv=cv, scoring=mse_scoring)
    cv_rfr.fit(X_train, y_train)
    best_params = cv_rfr.best_params_
    best_score = cv_rfr.best_score_
    print(best_params)
    print(best_score)

    n_estimators = best_params["n_estimators"]
    max_depth = best_params["max_depth"]
    max_features = best_params["max_features"]
    random_state = best_params["random_state"]
    min_samples_leaf = best_params["min_samples_leaf"]
    rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                random_state=random_state, min_samples_leaf=min_samples_leaf, oob_score=True)
    rfr.fit(X_train, y_train)

    y_rfr_train_pred = rfr.predict(X_train)
    y_rfr_test_pred = rfr.predict(X_test)
    # print(y_train)
    # print(y_rfr_train_pred)
    # print(y_rfr_train_pred)

    # TODO: Show variable importances

    ######################################### Evaluation ###############################################
    """ MSE and R2 """
    rfr_train_mse = mean_squared_error(y_train, y_rfr_train_pred)
    rfr_train_r2 = rfr.score(X_train, y_train)
    rfr_test_mse = mean_squared_error(y_test, y_rfr_test_pred)
    rfr_test_r2 = rfr.score(X_test, y_test)

    """ Compute the cross-validation scores """
    rfr_mse_scores = cross_val_score(
        rfr, X, y, cv=cv, scoring=mse_scoring)
    rfr_rmse_scores = cross_val_score(
        rfr, X, y, cv=cv, scoring=rmse_scoring)
    rfr_mse_mean_scores = abs(rfr_mse_scores.mean())
    rfr_std_scores = rfr_mse_scores.std()
    rfr_rmse_mean_scores = abs(rfr_rmse_scores.mean())
    rfr_rmse_std_scores = rfr_rmse_scores.std()

    # TODO: Get to know more about cross validation https://scikit-learn.org/stable/modules/cross_validation.html
    # TODO: Apply cross_score_predict ? https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict
    # TODO: Other metrics options: precision, recall, f1, etc. https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    ######################################### Visualization ###############################################
    plt.figure(figsize=(5, 5))
    plt.scatter(x=y_train, y=y_rfr_train_pred, c="#7CAE00", alpha=0.3)
    # A trend line to the plot
    z = np.polyfit(y_train, y_rfr_train_pred, 1)
    # print(z)
    p = np.poly1d(z)
    plt.plot(y_train, p(y_train), "#F8766D")
    plt.title('Using random forest regressor', fontsize=15)
    plt.ylabel('Predicted')
    plt.xlabel('Experimental')
    plt.show()

    ######################################### Conclusion of Evaluation ###############################################
    data = [
        ['Random forest',
         # rfr_train_mse, np.sqrt((rfr_train_mse)), rfr_train_r2,
         rfr_test_mse, np.sqrt((rfr_test_mse)), rfr_test_r2, rfr_mse_mean_scores, rfr_rmse_mean_scores, rfr_std_scores],
    ]
    results = pd.DataFrame(data,
                           columns=['Method',
                                    # 'Training MSE', 'Training RMSE', 'Training R2',
                                    'Test MSE', 'Test RMSE', 'Test R2', 'Cross-Val MSE Mean', 'Cross-Val RMSE Mean', 'Cross-Val MSE Standard Deviation']
                           )
    print(results)


# print('version: {}'.format(pd.__version__))

""" Ask user for input """
platform = input(
    "Enter the platform data to start (enter either `linux` or `macBook`): ")

""" Initialize """
multicollinearity_drop_columns = []

""" Load the csv file """
if (platform == "macBook"):
    # pid,name,username,memory_percent,cpu_percent,nice,create_time,mem_rss,mem_vms,mem_pfaults,mem_pageins,num_ctx_switches_voluntary,num_ctx_switches_involuntary,utime,stime,cutime,cstime
    df = pd.read_csv('../data_collecting/darwin_processes.csv')
    X_drop_columns = ['mem_pfaults']
    y_column = 'mem_pfaults'
elif (platform == "linux"):
    # PID, NAME, ELAPSED_TIME, TOTAL_TIME, utime, stime, start_time, uptime
    df = pd.read_csv('../data_collecting/linux_log_file.csv')
    X_drop_columns = ['stime']
    # TODO: Manually determined here
    multicollinearity_drop_columns = [
        "ELAPSED_TIME", "TOTAL_TIME", "uptime"]
    y_column = 'stime'
else:
    print("Invalid input")
    exit()

######################################### Data Preprocessing ###############################################
""" Check df """
# print(df.columns)
# print(df.dtypes)

""" Get obj cols and obj-col df """
obj_type_cols = [
    i for i in df.columns if df[i].dtype == 'object']
# Filter obj type columns
if (platform == "macBook"):
    obj_type_cols.append("pid")
elif (platform == "linux"):
    obj_type_cols.append("PID")
# print(obj_type_cols)
df_obj_type_cols = df[obj_type_cols]
# print(df_obj_type_cols)

""" Find the correlation among the columns """
corr = df.corr()
# print(corr)

""" Show correlation plot """
# plt.figure(figsize=(12, 4))
# sns.heatmap(corr, cmap='Wistia', annot=True)
# plt.title('Correlation in the dataset')
# plt.show()

""" ML algorithms cannot work with categorical data directly, categorical data must be converted to number.
# 1. Label Encoding
# 2. One hot encoding
# 3. Dummy variable trap
"""

# 1. Label Encoding
le = LabelEncoder()  # Code categories into 0,1,2.....
updated_df = df
for col in df_obj_type_cols:
    updated_df[col] = le.fit_transform(updated_df[col])
# print(updated_df.head())

# 2. One-Hot Encoding (Not best work in this case)
# ohe = OneHotEncoder()
# for col in df_obj_type_cols:
#     # Reshape the 1-D array to 2-D as fit_transform expects 2-D and finally fit the object
#     ohe_data = ohe.fit_transform(
#         updated_df[col].values.reshape(-1, 1)).toarray()
#     # Add this back into the original dataframe
#     ohe_df = pd.DataFrame(
#         ohe_data, columns=[col+"_"+str(int(i)) for i in range(ohe_data.shape[1])])
#     updated_df = pd.concat([updated_df, ohe_df], axis=1)
# # Drop the obj columns
# updated_df = updated_df.drop(df_obj_type_cols, axis=1)

# 3. Dummy variable trap (Not best work in this case)
# updated_df = pd.get_dummies(data=updated_df, prefix='OHE', prefix_sep='_',
#                            columns=obj_type_cols,
#                            drop_first=True,
#                            dtype='int8')

""" Drop un-necessary information such as all 0 in the same col """
for col in updated_df.columns:
    if (updated_df[col] == 0).all(axis=0):
        updated_df = updated_df.drop(col, axis=1)
# print(updated_df)

""" Check missing value """
# plt.figure(figsize=(12, 4))
# sns.heatmap(updated_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
# plt.title('Missing value in the dataset')
# plt.show()

""" Finish data preprocessing """
# print(df)
# print(updated_df)

""" Ask user for input """
model = input(
    "Enter the model to build (enter either `lr` for linear regression or `rfr` random forest regression): ")

# Show start time
print("Model starts building @ ", datetime.now())

""" Load the csv file """
if model == "lr":
    linear_regression()
elif model == "rfr":
    random_forest_regression()
else:
    print("Invalid input")
    exit()

# Show end time
print("Model finishs building @ ", datetime.now())
