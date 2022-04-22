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



representationLower = "entropy"
representationUpper = "Entropy"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_entropy = np.concatenate((x_train_1,x_train_2))
y_train_entropy = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_entropy, y_test_entropy = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_entropy, y_unpacked_entropy = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_entropy, y_packed_entropy = pickle.load(f)
print(x_train_entropy.shape)
print(y_train_entropy.shape)
print(x_test_entropy.shape)
print(y_test_entropy.shape)
print(x_unpacked_entropy.shape)
print(y_unpacked_entropy.shape)
print(x_packed_entropy.shape)
print(y_packed_entropy.shape)





representationLower = "functions"
representationUpper = "Functions"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_functions = np.concatenate((x_train_1,x_train_2))
y_train_functions = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_functions, y_test_functions = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_functions, y_unpacked_functions = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_functions, y_packed_functions = pickle.load(f)
print(x_train_functions.shape)
print(y_train_functions.shape)
print(x_test_functions.shape)
print(y_test_functions.shape)
print(x_unpacked_functions.shape)
print(y_unpacked_functions.shape)
print(x_packed_functions.shape)
print(y_packed_functions.shape)




representationLower = "hexdump"
representationUpper = "Hexdump"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_hexdump = np.concatenate((x_train_1,x_train_2))
y_train_hexdump = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_hexdump, y_test_hexdump = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_hexdump, y_unpacked_hexdump = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_hexdump, y_packed_hexdump = pickle.load(f)
print(x_train_hexdump.shape)
print(y_train_hexdump.shape)
print(x_test_hexdump.shape)
print(y_test_hexdump.shape)
print(x_unpacked_hexdump.shape)
print(y_unpacked_hexdump.shape)
print(x_packed_hexdump.shape)
print(y_packed_hexdump.shape)


representationLower = "relocs"
representationUpper = "Relocs"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_relocs = np.concatenate((x_train_1,x_train_2))
y_train_relocs = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_relocs, y_test_relocs = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_relocs, y_unpacked_relocs = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_relocs, y_packed_relocs = pickle.load(f)
print(x_train_relocs.shape)
print(y_train_relocs.shape)
print(x_test_relocs.shape)
print(y_test_relocs.shape)
print(x_unpacked_relocs.shape)
print(y_unpacked_relocs.shape)
print(x_packed_relocs.shape)
print(y_packed_relocs.shape)





representationLower = "sections"
representationUpper = "Sections"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_sections = np.concatenate((x_train_1,x_train_2))
y_train_sections = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_sections, y_test_sections = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_sections, y_unpacked_sections = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_sections, y_packed_sections = pickle.load(f)
print(x_train_sections.shape)
print(y_train_sections.shape)
print(x_test_sections.shape)
print(y_test_sections.shape)
print(x_unpacked_sections.shape)
print(y_unpacked_sections.shape)
print(x_packed_sections.shape)
print(y_packed_sections.shape)







representationLower = "symbols"
representationUpper = "Symbols"

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_symbols = np.concatenate((x_train_1,x_train_2))
y_train_symbols = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test_symbols, y_test_symbols = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked_symbols, y_unpacked_symbols = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed_symbols, y_packed_symbols = pickle.load(f)
print(x_train_symbols.shape)
print(y_train_symbols.shape)
print(x_test_symbols.shape)
print(y_test_symbols.shape)
print(x_unpacked_symbols.shape)
print(y_unpacked_symbols.shape)
print(x_packed_symbols.shape)
print(y_packed_symbols.shape)






pathTrain = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/Training/PCAtransformed-FamOrder.pickle"
f = open(pathTrain,"rb")
data,family,label = pickle5.load(f)
x_train_strings = []
y_train_strings = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_train_strings.append(data[i])
            y_train_strings.append(IndexToAssign)

x_train_strings = np.asarray(x_train_strings)
y_train_strings = np.asarray(y_train_strings)
print(x_train_strings.shape)
print(y_train_strings.shape)
pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/Testing/PCAtransformed-FamOrder.pickle"
f = open(pathTest,"rb")
data,family,label = pickle5.load(f)
x_test_strings = []
y_test_strings = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_test_strings.append(data[i])
            y_test_strings.append(IndexToAssign)
x_test_strings = np.asarray(x_test_strings)
y_test_strings = np.asarray(y_test_strings)
print(x_test_strings.shape)
print(y_test_strings.shape)
pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/UpxUnpacked/PCAtransformed-FamOrder.pickle"
f = open(pathTest,"rb")
data,family,label = pickle5.load(f)
x_unpacked_strings = []
y_unpacked_strings = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_unpacked_strings.append(data[i])
            y_unpacked_strings.append(IndexToAssign)
x_unpacked_strings = np.asarray(x_unpacked_strings)
y_unpacked_strings = np.asarray(y_unpacked_strings)
print(x_unpacked_strings.shape)
print(y_unpacked_strings.shape)
pathTest = "/media/ahmed/HDD/FamilyAnalysis/UPX/UPXStringPCA/Final/UpxPacked/PCAtransformed-FamOrder.pickle"
f = open(pathTest,"rb")
data,family,label = pickle5.load(f)
x_packed_strings = []
y_packed_strings = []
for fo in FO:
    indexToConsider = family.index(fo)
    IndexToAssign = FO.index(fo)
    for i in range(len(data)):
        if label[i] == indexToConsider:
            x_packed_strings.append(data[i])
            y_packed_strings.append(IndexToAssign)

x_packed_strings = np.asarray(x_packed_strings)
y_packed_strings = np.asarray(y_packed_strings)
print(x_packed_strings.shape)
print(y_packed_strings.shape)
print(x_train_strings.shape)
print(y_train_strings.shape)
print(x_test_strings.shape)
print(y_test_strings.shape)
print(x_unpacked_strings.shape)
print(y_unpacked_strings.shape)
print(x_packed_strings.shape)
print(y_packed_strings.shape)


x_train = np.concatenate((x_train_entropy,x_train_functions,x_train_hexdump,x_train_relocs,x_train_sections,x_train_symbols,x_train_strings),axis = 1)
y_train = y_train_entropy
x_test = np.concatenate((x_test_entropy,x_test_functions,x_test_hexdump,x_test_relocs,x_test_sections,x_test_symbols,x_test_strings),axis = 1)
y_test = y_test_entropy
x_unpacked = np.concatenate((x_unpacked_entropy,x_unpacked_functions,x_unpacked_hexdump,x_unpacked_relocs,x_unpacked_sections,x_unpacked_symbols,x_unpacked_strings),axis = 1)
y_unpacked = y_unpacked_entropy
x_packed = np.concatenate((x_packed_entropy,x_packed_functions,x_packed_hexdump,x_packed_relocs,x_packed_sections,x_packed_symbols,x_packed_strings),axis = 1)
y_packed = y_packed_entropy

print("data:")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_unpacked.shape)
print(y_unpacked.shape)
print(x_packed.shape)
print(y_packed.shape)




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
# f = open("../priorWorks/updatedData/models/CombinedBaselineLR","wb")
# pickle.dump(clf,f)

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
# f = open("../priorWorks/updatedData/models/CombinedBaselineRF","wb")
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
#     f = open("../priorWorks/updatedData/models/CombinedBaselineKNN"+str(neighbor),"wb")
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
# f = open("../priorWorks/updatedData/models/CombinedBaselineSVM","wb")
# pickle.dump(clf,f)


#
#
#
# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
# x_unpacked = np.reshape(x_unpacked,(x_unpacked.shape[0],x_unpacked.shape[1]))
# x_packed = np.reshape(x_packed,(x_packed.shape[0],x_packed.shape[1]))
# # create model
# model = Sequential()
# model.add(keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1:])))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Dense(256, activation='relu'))
# # model.add(keras.layers.Dropout(0.25))
# # model.add(keras.layers.Flatten())
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
# with open("../priorWorks/updatedData/models/CombinedBaselineDNN.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("../priorWorks/updatedData/models/CombinedBaselineDNN.h5")
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

filter = 8
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
with open("../priorWorks/updatedData/models/CombinedBaselineCNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../priorWorks/updatedData/models/CombinedBaselineCNN.h5")
print("Saved model to disk")
