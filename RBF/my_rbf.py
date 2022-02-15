# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

from tensorflow.keras.layers import Layer
from keras.initializers import Initializer, Constant

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy import stats
from sklearn.cluster import KMeans

from sklearn.datasets import fetch_california_housing

class CentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):  # constructor for kmeans class, X = x_train when calling

        self.X = X
        self.max_iter = max_iter

    # call function so I can call kmeansinitializer as a function
    def __call__(self, shape, dtype=None):

        n_centers = shape[1]    # shape[0] = 8, X[1] = N
        # print(shape[1])
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        # print(self.X.shape)     # (12384, 8)
        km.fit(self.X)      # creating N cluster centers that are 8 numbers long

        km_returned = km.cluster_centers_.T     # transpose so it is (8, N)
        return km_returned

# creating an RBF layer for our model
class RBFLayer(Layer):
    def __init__(self, neurons, gamma, initializer='uniform', **kwargs):  # constructor for the layer

        self.neurons = neurons
        self.gamma = K.cast_to_floatx(gamma)
        self.initializer = initializer

        super().__init__(**kwargs)

    def build(self, input_shape):

        # print(input_shape)      # (None, 8)
        # print(self.neurons)     # N (how many "neurons" are in the layer)
        # creating weights
        self.my_weights = self.add_weight(name='my_weights',
                                              # (      8       ,      N      )
                                          shape=(input_shape[1], self.neurons),
                                          initializer=self.initializer,
                                          trainable=True)

        super().build(input_shape)

    # creating the final radial basis function: exp(-gamma*SUM(x-weights)^2) and returning the output
    def call(self, inputs):

        # calculating the distance of the input from all the weights
        difference = K.expand_dims(inputs) - self.my_weights    # (None, 8, 1) - (8, N) = (None, 8, N)
        my_sum = K.sum(K.pow(difference, 2), axis=1)    # adding all the 8s to make N neurons with 1 number each
        out = K.exp(-1 * self.gamma * my_sum)

        return out      # (None, N)


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

# creating the model, first putting an RBF layer and then a simple MLP
model = Sequential()
rbflayer = RBFLayer(neurons=4000, gamma=1,
                    initializer=CentersKMeans(x_train))
model.add(rbflayer)
model.add(Dense(1, activation ='linear'))

# compiling the model, using mse as loss and different optimizers
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            metrics=['mean_squared_error'])

# fitting and evaluation of the model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=80,
                    validation_data=(x_test, y_test),
                    verbose=1)

test_loss, test_mse = model.evaluate(x_test, y_test, verbose=2)

# making final predictions
y_pred = model.predict(x_test)
y_pred = y_pred.reshape((-1,))

# print(rbflayer.get_weights())
model.summary()

print("--- %s seconds ---" % (time.time() - start_time))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0, 1)
plt.show()

# average distance between the observed data values and the predicted data values
rmse = mean_squared_error(y_test, y_pred, squared = False)     
r2 = r2_score(y_test, y_pred)
print('rmse:', rmse)
print('r2:', r2)
