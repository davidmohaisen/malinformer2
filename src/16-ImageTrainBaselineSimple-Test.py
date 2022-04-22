from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
from sklearn.utils import shuffle
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_test_data(size):
    pathMalware = "/media/ahmed/HDD/FamilyAnalysis/testing/mal2images/"
    x = []
    y = []
    f = open("/media/ahmed/HDD/FamilyAnalysis/testing/testingOverall-fam_malw.pickle","rb")
    labels = pickle.load(f)
    counter = -1
    for i in labels:
        counter += 1
        for j in labels[i]:
            if "fee73ed60b8f9bb106252ea0c8ab2a34" in j:
                continue
            pic = Image.open(pathMalware+j+".png")
            pic = pic.resize(size);
            pix = np.array(pic)
            x.append(pix)

            y.append(counter)

    countSamples = []
    currentSamples = []
    for i in range(counter+1):
        countSamples.append(y.count(i))
        currentSamples.append(0)


    x = np.asarray(x)
    y = np.asarray(y)

    x = x.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    # x, y = shuffle(x, y, random_state=0)


    xtest = []
    ytest = []
    for i in range(len(x)):
        # if currentSamples[y[i]] < 0.8*countSamples[y[i]]:
        currentSamples[y[i]]+= 1
        xtest.append(x[i])
        ytest.append(y[i])

    xtest = np.asarray(xtest)
    ytest = np.asarray(ytest)

    # print(xtest.shape)
    # print(ytest.shape)
    return (xtest, ytest)

def load_data(size):
    pathMalware = "../images/"
    x = []
    y = []
    f = open("../overall-fam_malw.pickle","rb")
    labels = pickle.load(f)
    counter = -1
    for i in labels:
        counter += 1
        for j in labels[i]:
            pic = Image.open(pathMalware+j+".png")
            pic = pic.resize(size);
            pix = np.array(pic)
            x.append(pix)

            y.append(counter)

    countSamples = []
    currentSamples = []
    for i in range(counter+1):
        countSamples.append(y.count(i))
        currentSamples.append(0)


    x = np.asarray(x)
    y = np.asarray(y)

    x = x.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    x, y = shuffle(x, y, random_state=0)

    xtrain = []
    ytrain = []
    # xtest = []
    # ytest = []
    for i in range(len(x)):
        # if currentSamples[y[i]] < 0.8*countSamples[y[i]]:
        currentSamples[y[i]]+= 1
        xtrain.append(x[i])
        ytrain.append(y[i])
        # else:
        #     currentSamples[y[i]]+= 1
        #     xtest.append(x[i])
        #     ytest.append(y[i])
    xtrain = np.asarray(xtrain)
    ytrain = np.asarray(ytrain)
    # xtest = np.asarray(xtest)
    # ytest = np.asarray(ytest)
    print(xtrain.shape)
    print(ytrain.shape)
    # print(xtest.shape)
    # print(ytest.shape)
    return (xtrain, ytrain)#, (xtest, ytest)

# Load the CIFAR10 data.
(x_train, y_train) = load_data(size=(64,64))
(x_test, y_test) = load_test_data(size=(64,64))
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = Sequential()
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',optimizer=opt , metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))#, shuffle=True)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#
# model_json = model.to_json()
# with open("../Model/Image/Baseline.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../Model/Image/Baseline.h5")
# print("Saved model to disk")
