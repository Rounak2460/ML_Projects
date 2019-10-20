import numpy as np
import scipy as sp
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/My Drive/train.csv',header=None,delimiter=' ').values

X_train=data[:,:-2]
y1=data[:,-1]
y2=data[:,-2]
X_train = X_train.reshape(len(X_train),3,32,32).transpose([0,2, 3, 1])/255

from keras.utils import to_categorical
y_train = to_categorical(y1)

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
model = tf.keras.Sequential()
tf.compat.v1.random.set_random_seed(3)

model.add(layers.Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu', input_shape = (32,32,3)))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256,activation = 'relu'))
model.add(layers.Dense(100,activation = 'softmax'))

model.compile(optimizer= 'Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, validation_split =0.1, epochs=200, batch_size=1000, callbacks = [callback])
