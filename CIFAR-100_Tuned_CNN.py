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


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
it = datagen.flow(X_train, y_train)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)


from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,\
    BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from keras.models import Sequential

model=Sequential()

model.add(Conv2D(96,(3,3),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(Conv2D(96,(3,3),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(MaxPooling2D((3, 3),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(192,(3,3),padding='same',activation='relu'))
model.add(Conv2D(192,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(192,(3,3),padding='same',activation='relu'))


model.add(MaxPooling2D((3, 3),strides=2))




model.add(Conv2D(192,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.25))


model.add(Conv2D(192,(1,1),padding='same',activation = 'relu'))
model.add(Conv2D(100,(1,1),padding='same',activation = 'relu'))


model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dense(100,activation='softmax'))

from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
from keras import optimizers

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = datagen.flow(X_train[:45000], y_train[:45000], batch_size=64)
validation_generator = datagen.flow(X_train[45000:50000], y_train[40000:50000], batch_size=64)
history1 = model.fit_generator(train_generator, steps_per_epoch=len(X_train[:40000]) /64, epochs=70, validation_data=validation_generator, validation_steps= 10000//64)
