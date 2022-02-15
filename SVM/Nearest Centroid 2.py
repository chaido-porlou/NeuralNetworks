# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist

start_time = time.time()

# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# flattening image (pixel vector)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# convertion to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape, x_test.shape)

# normalization of pixel values from [0, 255] to [0, 1]
x_train, x_test = x_train / 255., x_test / 255.

# keeping only 2 classes, horse and dog
indices = np.where(y_train == 5)
x_train_1 = x_train[indices]
y_train_1 = y_train[indices]
indices = np.where(y_test == 5)
x_test_1 = x_test[indices]
y_test_1 = y_test[indices]

indices = np.where(y_train == 7)
x_train_2 = x_train[indices]
y_train_2 = y_train[indices]
indices = np.where(y_test == 7)
x_test_2 = x_test[indices]
y_test_2 = y_test[indices]

x_train = np.concatenate((x_train_1[:len(x_train_1)], x_train_2[:len(x_train_2)]))
y_train = np.concatenate((y_train_1[:len(y_train_1)], y_train_2[:len(y_train_2)]))
x_test = np.concatenate((x_test_1[:len(x_test_1)], x_test_2[:len(x_test_2)]))
y_test = np.concatenate((y_test_1[:len(y_test_1)], y_test_2[:len(y_test_2)]))

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = NearestCentroid()  # nearest centroid algorithm

# training of model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)  # prediction for all data
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # accuracy print

print("--- %s seconds ---" % (time.time() - start_time))

# prediction of 10 images from test set
n_images = 10
test_images = x_test[:n_images]
predictions = model.predict(test_images)

strings = []
strings.append("dog")
strings.append("horse")

# display of image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [32, 32, 3]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % predictions[i])
    print(strings[predictions[i]])
