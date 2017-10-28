# import the necessary packages
from __future__ import print_function
from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1671)

# define the convnet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=5, padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        #add a softmax layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

# metwork and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10 # NUMBER OF OUTPUTS = NUMBER OF DIGITS
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
#data: shuffle and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
k.set_image_dim_ordering("th")
# consider then as float and normalize them
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape[0], 'train samples')
print(X_test.shape[1], 'test samples')
# convert class vector to binary class matrices
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)
# initialize the model and accuracy
model = LeNet.build(INPUT_SHAPE, NB_CLASSES)
print('!!!!!!!!!!!!!!!!!!!!!!!! printing the model !!!!!!!!!!!!!!!!!!!!!!!!')
model.summary()
print('!!!!!!!!!!!!!!!!!!!!!!!! printing the model !!!!!!!!!!!!!!!!!!!!!!!!')
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
              metrics=['accuracy'])