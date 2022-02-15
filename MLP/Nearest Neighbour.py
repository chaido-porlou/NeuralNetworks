# -*- coding: utf-8 -*-

from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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

model = KNeighborsClassifier(n_neighbors=3)  # chose n=1 & n=3

# training of model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)  # prediction for all data
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # acuracy print

print("--- %s seconds ---" % (time.time() - start_time))

# prediction of 10 images from test set
n_images = 10
test_images = x_test[:n_images]
predictions = model.predict(test_images)

strings = []
strings.append("airplane")
strings.append("automobile")
strings.append("bird")
strings.append("cat")
strings.append("deer")
strings.append("dog")
strings.append("frog")
strings.append("horse")
strings.append("ship")
strings.append("truck")

# display of image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [32, 32, 3]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % predictions[i])
    print(strings[predictions[i]])
