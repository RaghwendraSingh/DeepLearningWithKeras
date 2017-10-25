from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
np.random.seed(1678)

REPARTITIONED = 784
NB_CLASSES = 10
N_HIDDEN = 128
BATCH__SIZE = 128
EPOCHS = 23
DROPOUT = 0.3
OPTIMIZER = Adam()
VERBOSE = 1
VALIDATION_SPLIT = 0.2


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#preprocessing
X_train = X_train.reshape(60000, REPARTITIONED)
X_test = X_test.reshape(10000, REPARTITIONED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#normalizing
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(REPARTITIONED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
            metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH__SIZE,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print('\nTest score: ', score[0])
print('\nTest accuracy: ', score[1])