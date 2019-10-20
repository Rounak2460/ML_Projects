import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
data = pd.read_csv("train.csv",header = None, delimiter = ' ', nrows = 50000).values
test_data = pd.read_csv("test.csv",header = None, delimiter = ' ', nrows = 10000).values
train_x = data[:,0:len(data[0])-1]
test_x = test_data[:,0:len(test_data[0])-1]
train_x = train_x.reshape(len(train_x),3,1024)
test_x = test_x.reshape(len(test_x),3,1024)
a = np.zeros((len(train_x),1024,3))


for i in range(len(train_x)):
    z = train_x[i][0].reshape(1024,1)
    for j in range(2):
        z = np.concatenate((z,train_x[i][j+1].reshape(1024,1)), axis = 1)
    a[i] = z

train_x = a.reshape(len(train_x),32,32,3)
train_x /= 255.0

b = np.zeros((len(test_x),1024,3))
for i in range(len(test_x)):
    u = test_x[i][0].reshape(1024,1)
    for j in range(2):
        u = np.concatenate((u,test_x[i][j+1].reshape(1024,1)), axis = 1)
    b[i] = u

test_x = b.reshape(len(test_x),32,32,3)
test_x /= 255.0

train_y = data[0:50000,len(data[0])-1:len(data[0])]
train_y = to_categorical(train_y)

model = tf.keras.Sequential()
tf.compat.v1.random.set_random_seed(3)

model.add(layers.Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu', input_shape = (32,32,3)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (1,1), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (1,1), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(layers.Flatten())

model.add(layers.Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(10,activation = 'softmax'))



model.compile(optimizer= 'Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

output = open("pred.txt", "w")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_x, train_y, validation_split =0.1, epochs=50, batch_size=100, callbacks = [callback])
prediction = model.predict(test_x)
Y = ((np.argmax(prediction, axis = 1))).reshape((10000,1))
print(Y.shape)
np.savetxt(output, Y)
train_loss, train_acc = model.evaluate(train_x, train_y)
print('Train accuracy:', train_acc)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
