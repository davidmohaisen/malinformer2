import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential

x = []
y = []
varience = "0.999"
label = -1
for i in np.arange(0,1000,100):
    label += 1
    f = open("../pca/PCA2components-family_variance-"+str(i)+"-"+varience+".pickle","rb")
    Data,_ = pickle.load(f)
    for j in range(len(Data)):
        x.append(Data[j])
        y.append(label)
x = np.asarray(x)
y = np.asarray(y)
x = np.reshape(x,(x.shape[0],x.shape[1],1))
y = keras.utils.to_categorical(y, 10)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = [],[],[],[]

total = 100
subcurrent = []
for i in range(10):
    subcurrent.append(0)
for i in range(len(x)):
    if subcurrent[np.argmax(y[i])] < 0.8*total:
        subcurrent[np.argmax(y[i])]+= 1
        x_train.append(x[i])
        y_train.append(y[i])
    else:
        subcurrent[np.argmax(y[i])]+= 1
        x_test.append(x[i])
        y_test.append(y[i])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

filter = 64
# create model
model = Sequential()
model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv1D(filter*3,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter*3,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# Fit the model\
model.fit(x_train,y_train, epochs=100, batch_size=8)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
