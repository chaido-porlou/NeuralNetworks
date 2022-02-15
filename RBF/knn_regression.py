# -*- coding: utf-8 -*-
import numpy as np
import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy import stats

from sklearn.datasets import fetch_california_housing
from sklearn import neighbors

start_time = time.time()

x_train, y_train = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.40)

# convertion to float32
x_train = np.array(x_train, np.float32)
y_train = np.array(y_train, np.float32)

# z-score normalization for better results
x_train = stats.zscore(x_train)
y_train = stats.zscore(y_train)
x_test = stats.zscore(x_test)
y_test = stats.zscore(y_test)

start_time = time.time()

# using the knn algorithm for regression
model = neighbors.KNeighborsRegressor(n_neighbors = 5)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

# average distance between the observed data values and the predicted data values
rmse = mean_squared_error(y_test, y_pred, squared = False)     
r2 = r2_score(y_test, y_pred)
print('rmse:', rmse)
print('r2:', r2)

print("--- %s seconds ---" % (time.time() - start_time))
