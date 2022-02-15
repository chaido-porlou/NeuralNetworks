from __future__ import print_function

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, mnist

start_time = time.time()

# model building
model = Sequential()
# one VGG block because convolution helps with accuracy, but not more so the model stays simple
model.add(Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform')) # tried with 64, 128 neurons
# tried with a = 0.01, 0.001
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(Dropout(0.5)) # tried with 0.2, 0.3, 0.5 and no dropout
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform')) #tried with 128, 256 neurons
# tried with a = 0.01, 0.001
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(Dropout(0.5))  # tried with 0.2, 0.3, 0.5 and no dropout
model.add(Dense(10, activation='softmax'))

# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# convertion to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# normalization of pixel values from [0, 255] to [0, 1]
x_train, x_test = x_train / 255., x_test / 255.

opt = tf.keras.optimizers.SGD(learning_rate=0.01) # chose SGD optimizer with lr = 0.01 (tried with 0.1, 0.001 in final model)
# compilation of model
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training of model 
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, 
                    batch_size=64) # chose 50 epochs (for time) and batch size = 64

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) # accurate accuracy
print('\nTest accuracy:', test_acc)

print("--- %s seconds ---" % (time.time() - start_time))

# prediction of 40 images from test set
n_images = 40
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
    print("Model prediction: %i" % np.argmax(predictions[i]))
    print(strings[np.argmax(predictions[i])])

model.summary()

# diagram for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# diagram for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# prediction and editing of all data for classification report
y_pred = model.predict(x_test)
y_pred_1 = list()
for i in range(10000):
    y_pred_1.append(np.argmax(y_pred[i]))
    true_labels = np.array(y_pred_1)

y_test_labels = y_test.reshape(-1,1)
y_pred_labels  = true_labels.reshape(-1,1)

print(classification_report(y_test_labels, y_pred_labels, target_names=strings))
