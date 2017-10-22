from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

RESHAPE = 784
NB_CLASSES = 10
N_HIDDEN = 128
OPTIMIZER = SGD()
VERBOSE = 1
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
NB_EPOCHS = 20
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# preprocessing data

X_train = X_train.reshape(60000, RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# notemalizing
X_train /= X_train
X_test /= X_test

# convert class labels to binary class matrices
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])


history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('Test score', score[0])
print('Test accuracy', score[1])