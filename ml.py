import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

LABEL_DATA_COL_INDEX = [-3, -2, -1]

# Mean Absolute Percentage Error
# https://brunch.co.kr/@chris-song/34
def MAPE(y_test, y_hat):
    return np.mean(np.abs((y_test - y_hat) / y_test)) * 100


def ml():
    # refer to sample_dataset_columns.txt for column numbers.
    df = pd.read_csv("sample_dataset_5000.csv", header=None)

    # Exclude columns that contain only zeros.
    # https://stackoverflow.com/a/21165116
    # df = df.loc[:, (df != 0).any(axis=0)]

    # Exclude both columns of src host cpu and dst host cpu because unexpected some measurements errors of always 100.
    # TODO: collect a new dataset
    # df = df.drop(columns=[50, 58])

    # Include a minimal feature set only (generally considered as most relevant to the live migration performance).
    # note that iloc is 0-base index and the following collections were averaged during the profiling interval in _do_profiling.
    # 0: vm___swap______swap___cached___value___gauge: current size of swap spaces that are cached (KB)
    # 2: vm___swap______swap___used___value___gauge: current size of swap spaces that are used (KB). exclusive to cached swap
    # 7: vm___vmem______vmpage_faults______majflt___derive: major pagefault rate
    # 8: vm___vmem______vmpage_faults______minflt___derive: minor pagefault rate
    # 18: vm___vmem______vmpage_number___dirty___value___gauge: current number of pages that are dirty
    # 41: vm___vmem______vmpage_number___writeback___value___gauge: current number of pages that are under writeback
    # 50: src.host___cpu_usage___value___gauge
    # 56: src.host___eno1___if_packets___tx___derive
    # 58: dst.host___cpu_usage___value___gauge
    # 65: dst.host___eno1___if_packets___rx___derive
    min_features = [0, 2, 7, 8, 18, 41, 50, 56, 58, 65]
    df = df.iloc[:, min_features + LABEL_DATA_COL_INDEX]

    print(df.shape)
    print(df.describe())
    # print(df.hist(bins=20, figsize=(20, 15)))
    # plt.show()

    # X: feature data (multiple values)
    X = df.iloc[:, :-len(LABEL_DATA_COL_INDEX)].values
    # X = df.iloc[:, 0:10].values   # using first 10 features only

    # Y: label data (single value). uncomment the corresponding line for label you want to predict
    #y = df.iloc[:, -3:-2].values      # label: total migration time (MT)
    #y = df.iloc[:, -2:-1].values        # label: packet loss count that can derive service downtime (DT) by multiplying ping interval
    y = df.iloc[:, -1:].values        # label: vm downtime
    print(y)

    # Note: risk of information leak from train to test set. read the docs
    # A common mistake is to apply it to the entire data before splitting into training and test sets.
    # ref: https://sebastianraschka.com/faq/docs/scale-training-test.html
    # X = preprocessing.scale(X)

    # Fix random_state to 1 (a seed val.) to always get the same record distribution both for train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # 1. the best practice use of this scaler is to fit it on the training dataset
    # and then apply the transform to the training dataset, and the test dataset.
    # ref: https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
    # 2. which scaler is used? it depends on the distribution of the dataset...
    # ref: https://mkjjo.github.io/python/2019/01/10/scaler.html
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Flatten the y-values into 1d array.
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # 1. Linear Regression
    lr = LinearRegression(fit_intercept=True, n_jobs=None)
    start_time = time.perf_counter()
    lr.fit(X_train_scaled, y_train)
    learning_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    y_hat = lr.predict(X_test_scaled)
    prediction_time = time.perf_counter() - start_time
    print("\n=====Linear Regression=====")
    # print(y_hat)
    # Quantifying the quality of predictions in Regression problem
    # https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    # Type 1. scale-dependent errors
    print("MAE: ", mean_absolute_error(y_test, y_hat))
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
    # Type 2. percentage errors
    # print("MAPE: ", MAPE(y_test, y_hat))
    print("MAPE: ", 100 * mean_absolute_percentage_error(y_test, y_hat))
    # Type 3. R² score, the coefficient of determination
    print("R-squared: ", r2_score(y_test, y_hat))
    print("Learning time (s): ", f'{learning_time:.5f}')
    print("Prediction time (s): ", f'{prediction_time:.5f}')

    # TODO: the reference paper uses 10-fold cross validation. How about using PipeLine?
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # https://stackoverflow.com/questions/52249158/do-i-give-cross-val-score-the-entire-dataset-or-just-the-training-set
    # print("Avg. score/accuracy:", cross_val_score(lr, X_train_scaled, y_train, cv=10))

    # 2. Support Vector Regression with polynomial kernel function
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    svr_poly = SVR(kernel='poly', degree=10, C=10.0)
    start_time = time.perf_counter()
    svr_poly.fit(X_train_scaled, y_train)
    learning_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    y_hat = svr_poly.predict(X_test_scaled)
    prediction_time = time.perf_counter() - start_time
    print("\n=====SVR w/ polynomial kern=====")
    # print(y_hat)
    print("MAE:", mean_absolute_error(y_test, y_hat))
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
    # print("MAPE:", MAPE(y_test, y_hat))
    print("MAPE: ", 100 * mean_absolute_percentage_error(y_test, y_hat))
    print("R-squared: ", r2_score(y_test, y_hat))
    print("Learning time (s): ", f'{learning_time:.5f}')
    print("Prediction time (s): ", f'{prediction_time:.5f}')

    # 3. Support Vector Regression with Radial Basis Function kernel
    # Note: the reference paper uses rbf w/ C=10 and standardized features
    # C=100.0 results in ~0.8 R2 score
    # svr_rbf = SVR(kernel='rbf', C=10.0)
    # start_time = time.perf_counter()
    # svr_rbf.fit(X_train_scaled, y_train)
    # learning_time = time.perf_counter() - start_time
    # start_time = time.perf_counter()
    # y_hat = svr_rbf.predict(X_test_scaled)
    # prediction_time = time.perf_counter() - start_time
    # print("\n=====SVR w/ radial basis function kern=====")
    # # print(y_hat)
    # print("MAE:", mean_absolute_error(y_test, y_hat))
    # print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
    # # print("MAPE:", MAPE(y_test, y_hat))
    # print("MAPE: ", 100 * mean_absolute_percentage_error(y_test, y_hat))
    # print("R-squared: ", r2_score(y_test, y_hat))
    # print("Learning time (s): ", f'{learning_time:.5f}')
    # print("Prediction time (s): ", f'{prediction_time:.5f}')

    # 4. SVR w/ rbf kern in Bootstrap Aggregation mode (ensemble-based Bagging regressor)
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
    regr = BaggingRegressor(base_estimator=SVR(kernel='rbf', C=10.0), n_estimators=64, bootstrap=True)
    start_time = time.perf_counter()
    regr.fit(X_train_scaled, y_train)
    learning_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    y_hat = regr.predict(X_test_scaled)
    prediction_time = time.perf_counter() - start_time
    print("\n=====SVR w/ rbf kern in bootstrap aggr=====")
    # print(y_hat)
    print("MAE:", mean_absolute_error(y_test, y_hat))
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
    # print("MAPE:", MAPE(y_test, y_hat))
    print("MAPE: ", 100 * mean_absolute_percentage_error(y_test, y_hat))
    print("R-squared: ", r2_score(y_test, y_hat))
    print("Learning time (s): ", f'{learning_time:.5f}')
    print("Prediction time (s): ", f'{prediction_time:.5f}')

    # 5. SVR w/ linear kern in Bootstrap Aggregation mode
    # regr2 = BaggingRegressor(base_estimator=SVR(kernel='linear', C=50.0), n_estimators=64, bootstrap=True)
    # # regr2 = BaggingRegressor(base_estimator=LinearRegression(fit_intercept=True, n_jobs=None),
    # #                          n_estimators=64, bootstrap=True)
    # start_time = time.perf_counter()
    # regr2.fit(X_train_scaled, y_train)
    # learning_time = time.perf_counter() - start_time
    # start_time = time.perf_counter()
    # y_hat = regr2.predict(X_test_scaled)
    # prediction_time = time.perf_counter() - start_time
    # print("\n=====SVR w/ linear kern in bootstrap aggr=====")
    # # print(y_hat)
    # print("MAE:", mean_absolute_error(y_test, y_hat))
    # print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_hat)))
    # # print("MAPE:", MAPE(y_test, y_hat))
    # print("MAPE: ", 100 * mean_absolute_percentage_error(y_test, y_hat))
    # print("R-squared: ", r2_score(y_test, y_hat))
    # print("Learning time (s): ", f'{learning_time:.5f}')
    # print("Prediction time (s): ", f'{prediction_time:.5f}')


if __name__ == '__main__':
    ml()
