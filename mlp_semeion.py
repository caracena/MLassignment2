import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import  pandas as pd
import sklearn.cross_validation as cross_validation
import matplotlib.pyplot as plt

all_data = pd.read_csv('semeion.data', header=None,sep=' ')
x = all_data.ix[:,:255].values
y = all_data.ix[:,256:265].values

X_train,X_test,y_train,y_test = cross_validation.train_test_split(x, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(256, input_shape=(256,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=10, nb_epoch=1000,
                    verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()

print('Test score:', score[0])
print('Test accuracy:', score[1])