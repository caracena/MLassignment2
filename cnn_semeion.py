import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import  pandas as pd
import sklearn.cross_validation as cross_validation
import matplotlib.pyplot as plt

batch_size = 10
nb_classes = 10
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 16, 16
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

all_data = pd.read_csv('semeion.data', header=None,sep=' ')
x = all_data.ix[:,:255].values
y = all_data.ix[:,256:265].values

X_train,X_test,y_train,y_test = cross_validation.train_test_split(x, y, test_size=0.33, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()