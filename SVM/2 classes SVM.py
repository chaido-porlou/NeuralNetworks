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
from sklearn import svm

different_c = [0.001, 0.01, 0.1, 1, 10, 100] # values for different c
different_gamma = [0.001, 0.01, 0.1, 1, 10] # values for different gamma

# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# flattening image (pixel vector)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# convertion to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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

for c in different_c: # different_gamma also used

    start_time = time.time()

    print(c)
    # creation of svm classifier
    clf = svm.SVC(kernel='poly', C = c, gamma = 0.001, degree=2, verbose=True) # kernel
    # clf = svm.LinearSVC(C = c, max_iter = 100000) # used but didn't keep 

    # training of the model using the training set
    clf.fit(x_train, y_train)

    # prediction of testing set
    y_pred = clf.predict(x_test)
    pred = clf.predict(x_train)
    train_acc = np.mean(pred == y_train)

    print("Training Accuracy:", train_acc)
    print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred))

    print("--- %s seconds ---" % (time.time() - start_time))

    # prediction of 10 images from test set
    random_indices = np.random.choice(x_test.shape[0], size=10, replace=False)
    test_images = x_test[random_indices]

    strings = []
    strings.append("dog")
    strings.append("horse")

    # display of image and model prediction
    j = 0
    for i in random_indices:
        plt.imshow(np.reshape(test_images[j], [32, 32, 3]), cmap='gray')
        plt.show()
        j = j + 1
        print("Model prediction: %i" % y_pred[i])
        if y_pred[i] == 5:
            print(strings[0])
        else:
            print(strings[1])


    # getting support vectors themselves (and their number)
    support_vectors = clf.support_vectors_

    # visualizing support vectors (did not use after all)
    plt.scatter(x_train[:,0], x_train[:,1])
    plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
    plt.title('Linearly separable data with support vectors')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
