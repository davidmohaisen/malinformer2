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
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import pickle5
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

representationLower = "strings"
representationUpper = "Strings"






pathTrain = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/Training/PCAtransformed-FamOrder.pickle"
f = open(pathTrain,"rb")
data,family,label = pickle5.load(f)



x_train = []
y_train = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_train.append(data[i])
            y_train.append(IndexToAssign)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)
print(y_train.shape)




pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/Testing/PCAtransformed-FamOrder.pickle"
f = open(pathTest,"rb")
data,family,label = pickle5.load(f)



x_test = []
y_test = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_test.append(data[i])
            y_test.append(IndexToAssign)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print(x_test.shape)
print(y_test.shape)





pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/UpxUnpacked/PCAtransformed-FamOrder.pickle"
f = open(pathTest,"rb")
data,family,label = pickle5.load(f)



x_unpacked = []
y_unpacked = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_unpacked.append(data[i])
            y_unpacked.append(IndexToAssign)

x_unpacked = np.asarray(x_unpacked)
y_unpacked = np.asarray(y_unpacked)
print(x_unpacked.shape)
print(y_unpacked.shape)




pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/UpxPacked/PCAtransformed-FamOrder.pickle"


f = open(pathTest,"rb")
data,family,label = pickle5.load(f)

x_packed = []
y_packed = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_packed.append(data[i])
            y_packed.append(IndexToAssign)

x_packed = np.asarray(x_packed)
y_packed = np.asarray(y_packed)
print(x_packed.shape)
print(y_packed.shape)






print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_unpacked.shape)
print(y_unpacked.shape)
print(x_packed.shape)
print(y_packed.shape)


#
#
# print("==========================================================================")
# print("LR")
# print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
#
# score = clf.score(x_test, y_test)
# print("The LR score is:",score)
# y_true = y_test
# y_pred = clf.predict(x_test)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# score = clf.score(x_unpacked, y_unpacked)
# print("The LR score unpacked is:",score)
# y_true = y_unpacked
# y_pred = clf.predict(x_unpacked)
# print(f1_score(y_true, y_pred, average='weighted'))
#
#
# score = clf.score(x_packed, y_packed)
# print("The LR score packed is:",score)
# y_true = y_packed
# y_pred = clf.predict(x_packed)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# f = open("../priorWorks/updatedData/models/"+representationUpper+"BaselineLR","wb")
# pickle.dump(clf,f)
#
#
# print("==========================================================================")
# print("RF")
# print("==========================================================================")
# clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
#
# score = clf.score(x_test, y_test)
# print("The RF score is:",score)
# y_true = y_test
# y_pred = clf.predict(x_test)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# score = clf.score(x_unpacked, y_unpacked)
# print("The RF score unpacked is:",score)
# y_true = y_unpacked
# y_pred = clf.predict(x_unpacked)
# print(f1_score(y_true, y_pred, average='weighted'))
#
#
# score = clf.score(x_packed, y_packed)
# print("The RF score packed is:",score)
# y_true = y_packed
# y_pred = clf.predict(x_packed)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# f = open("../priorWorks/updatedData/models/"+representationUpper+"BaselineRF","wb")
# pickle.dump(clf,f)
#
# print("==========================================================================")
# print("KNN")
# print("==========================================================================")
# neighbors = [3,5,10]
# for neighbor in neighbors:
#     clf = KNeighborsClassifier(n_neighbors=neighbor).fit(x_train, y_train)
#
#     score = clf.score(x_test, y_test)
#     print("The KNN "+str(neighbor)+" score is:",score)
#     y_true = y_test
#     y_pred = clf.predict(x_test)
#     print(f1_score(y_true, y_pred, average='weighted'))
#
#     score = clf.score(x_unpacked, y_unpacked)
#     print("The KNN "+str(neighbor)+" score unpacked is:",score)
#     y_true = y_unpacked
#     y_pred = clf.predict(x_unpacked)
#     print(f1_score(y_true, y_pred, average='weighted'))
#
#
#     score = clf.score(x_packed, y_packed)
#     print("The KNN "+str(neighbor)+" score packed is:",score)
#     y_true = y_packed
#     y_pred = clf.predict(x_packed)
#     print(f1_score(y_true, y_pred, average='weighted'))
#
#     f = open("../priorWorks/updatedData/models/"+representationUpper+"BaselineKNN"+str(neighbor),"wb")
#     pickle.dump(clf,f)
#
# from sklearn.svm import SVC
# print("==========================================================================")
# print("SVM")
# print("==========================================================================")
# clf = SVC().fit(x_train, y_train)
#
# score = clf.score(x_test, y_test)
# print("The SVM score is:",score)
# y_true = y_test
# y_pred = clf.predict(x_test)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# score = clf.score(x_unpacked, y_unpacked)
# print("The SVM score unpacked is:",score)
# y_true = y_unpacked
# y_pred = clf.predict(x_unpacked)
# print(f1_score(y_true, y_pred, average='weighted'))
#
#
# score = clf.score(x_packed, y_packed)
# print("The SVM score packed is:",score)
# y_true = y_packed
# y_pred = clf.predict(x_packed)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# f = open("../priorWorks/updatedData/models/"+representationUpper+"BaselineSVM","wb")
# pickle.dump(clf,f)





# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# x_unpacked = np.reshape(x_unpacked,(x_unpacked.shape[0],x_unpacked.shape[1],1))
# x_packed = np.reshape(x_packed,(x_packed.shape[0],x_packed.shape[1],1))
# # create model
# model = Sequential()
# model.add(keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1:])))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Dense(256, activation='relu'))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# # model.add(keras.layers.Dense(256, activation='relu'))
# # model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(10, activation="softmax"))
#
# # Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=10, batch_size=16)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test accuracy:', scores[1])
# y_true = y_test
# y_pred = np.argmax(model.predict(x_test),axis=1)
# print(f1_score(y_true, y_pred, average='weighted'))
#
#
# scores = model.evaluate(x_unpacked, y_unpacked, verbose=1)
# print('Unpacked accuracy:', scores[1])
# y_true = y_unpacked
# y_pred = np.argmax(model.predict(x_unpacked),axis=1)
# print(f1_score(y_true, y_pred, average='weighted'))
#
# scores = model.evaluate(x_packed, y_packed, verbose=1)
# print('Packed accuracy:', scores[1])
# y_true = y_packed
# y_pred = np.argmax(model.predict(x_packed),axis=1)
# print(f1_score(y_true, y_pred, average='weighted'))
#
#
# model_json = model.to_json()
# with open("../priorWorks/updatedData/models/"+representationUpper+"BaselineDNN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../priorWorks/updatedData/models/"+representationUpper+"BaselineDNN.h5")
# print("Saved model to disk")


print("==========================================================================")
print("CNN")
print("==========================================================================")
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_unpacked = np.reshape(x_unpacked,(x_unpacked.shape[0],x_unpacked.shape[1],1))
x_packed = np.reshape(x_packed,(x_packed.shape[0],x_packed.shape[1],1))
# create model
model = Sequential()

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
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10, batch_size=16)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])
y_true = y_test
y_pred = np.argmax(model.predict(x_test),axis=1)
print(f1_score(y_true, y_pred, average='weighted'))


scores = model.evaluate(x_unpacked, y_unpacked, verbose=1)
print('Unpacked accuracy:', scores[1])
y_true = y_unpacked
y_pred = np.argmax(model.predict(x_unpacked),axis=1)
print(f1_score(y_true, y_pred, average='weighted'))

scores = model.evaluate(x_packed, y_packed, verbose=1)
print('Packed accuracy:', scores[1])
y_true = y_packed
y_pred = np.argmax(model.predict(x_packed),axis=1)
print(f1_score(y_true, y_pred, average='weighted'))



model_json = model.to_json()
with open("../priorWorks/updatedData/models/"+representationUpper+"BaselineCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../priorWorks/updatedData/models/"+representationUpper+"BaselineCNN.h5")
print("Saved model to disk")
