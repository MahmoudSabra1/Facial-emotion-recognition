import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras.backend as K
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, InceptionV3, Xception
import h5py

from resnets_utils import *

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# Preprocess data
data = pd.read_csv('fer2013.csv')
labels = pd.read_csv('fer2013new.csv')

orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown',
                    'NF']

n_samples = len(data)
w = 48
h = 48

y = np.array(labels[orig_class_names]).argmax(axis=-1)
mask = y < orig_class_names.index('unknown')

X = np.zeros((n_samples, w, h))
for i in range(n_samples):
    X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape(h, w)

X = X[mask]
y = y[mask]

class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# Normalize image vectors
X = X / 255.0

# Split Data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.2)

# # Show images
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap='gray')
#     plt.xlabel(class_names[y_train[i]])
# plt.show()

# Convert training and test labels to one hot matrices
y_train = convert_to_one_hot(y_train, len(class_names)).T
y_test = convert_to_one_hot(y_test, len(class_names)).T
y_val = convert_to_one_hot(y_val, len(class_names)).T

x_train = x_train.reshape(len(x_train), h, w, 1)
x_test = x_test.reshape(len(x_test), h, w, 1)
x_val = x_val.reshape(len(x_val), h, w, 1)

print("X_train shape: " + str(x_train.shape))
print("Y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(x_test.shape))
print("Y_test shape: " + str(y_test.shape))
print("X_val shape: " + str(x_val.shape))
print("Y_val shape: " + str(y_val.shape))

epochs = 15
# model = vgg(input_shape=(h, w, 1), classes=len(class_names))
model = ResNet50(weights=None, input_shape=(h, w, 1), pooling='max', classes=len(class_names))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_data=(x_val, y_val), verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test)

# Plotting accuracy graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='upper left')

plt.show()
