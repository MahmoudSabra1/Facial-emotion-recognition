import sys, os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.utils import np_utils
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
from sklearn import model_selection

data = pd.read_csv('fer2013.csv')
labels = pd.read_csv('fer2013new.csv')

orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown',
                    'NF']

n_samples = len(data)
w = 48
h = 48

# add 'contempt' probabilities to 'neutral' ones
arr = np.array(labels[orig_class_names])
arr[:, 0] += arr[:, 7]
arr[:, 7] = 0
y = arr.argmax(axis=-1)
mask = y < orig_class_names.index('unknown')

y = np.array(labels[orig_class_names]) * 0.1
y = y[:, :-2]


X = np.zeros((n_samples, w, h))
for i in range(n_samples):
    X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape(h, w)

X = X[mask]
y = y[mask]

class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
X = X / 255.0

# Split Data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.111, random_state=42)

"""
y_train = tf.one_hot(y_train, len(class_names))
y_test = tf.one_hot(y_test, len(class_names))
y_val = tf.one_hot(y_val, len(class_names))
"""

x_train = x_train.reshape(len(x_train), h, w, 1)
x_test = x_test.reshape(len(x_test), h, w, 1)
x_val = x_val.reshape(len(x_val), h, w, 1)

# data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True)
datagen.fit(x_train)

model = Sequential()

# 1st conv block
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(x_train.shape[1:])))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.5))

# 2nd conv block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 3rd conv block
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

# 4th conv block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# 5th conv block
model.add(Conv2D(256, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=100
                    , validation_data=(x_val, y_val), verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='upper left')

plt.show()

# Plot loss graph
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()

"""
fer_json = model.to_json()
with open("Saved-Models/noContempt", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("noContempt.h5")

json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
"""