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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

f = open("../priorWorks/TrainData/symbols_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_train1 = np.asarray(data)
families = np.asarray(families)


y_train1 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train1.append(indexOfLabel)
    count += 1
y_train1 = np.asarray(y_train1)

f = open("../priorWorks/TestData/symbols_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_test1 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test1 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test1.append(indexOfLabel)
    count += 1
y_test1 = np.asarray(y_test1)


print(x_train1.shape)
print(y_train1.shape)
print(x_test1.shape)
print(y_test1.shape)


f = open("../priorWorks/PCADisassymbly.pickle","rb")
x_train2,y_train2,x_test2,y_test2 = pickle.load(f)

print(x_train2.shape)
print(y_train2.shape)
print(x_test2.shape)
print(y_test2.shape)


f = open("../priorWorks/TrainData/entropyListByMalwaretrainData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
x_train3 = np.asarray(data)
families = np.asarray(families)


y_train3 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train3.append(indexOfLabel)
    count += 1
y_train3 = np.asarray(y_train3)

f = open("../priorWorks/TestData/entropyListByMalwaretestData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
x_test3 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test3 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test3.append(indexOfLabel)
    count += 1
y_test3 = np.asarray(y_test3)


print(x_train3.shape)
print(y_train3.shape)
print(x_test3.shape)
print(y_test3.shape)





f = open("../priorWorks/TrainData/functions_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_train4 = np.asarray(data)
families = np.asarray(families)


y_train4 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train4.append(indexOfLabel)
    count += 1
y_train4 = np.asarray(y_train4)

f = open("../priorWorks/TestData/functions_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_test4 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test4 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test4.append(indexOfLabel)
    count += 1
y_test4 = np.asarray(y_test4)


print(x_train4.shape)
print(y_train4.shape)
print(x_test4.shape)
print(y_test4.shape)




f = open("../priorWorks/TrainData/hexdumpJsonstrainData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
x_train5 = np.asarray(data)
print(len(data))
families = np.asarray(families)


y_train5 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train5.append(indexOfLabel)
    count += 1
y_train5 = np.asarray(y_train5)

f = open("../priorWorks/TestData/hexdumpJsonstestData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
x_test5 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test5 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test5.append(indexOfLabel)
    count += 1
y_test5 = np.asarray(y_test5)


print(x_train5.shape)
print(y_train5.shape)
print(x_test5.shape)
print(y_test5.shape)



f = open("../priorWorks/TrainData/relocs_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_train6 = np.asarray(data)
families = np.asarray(families)


y_train6 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train6.append(indexOfLabel)
    count += 1
y_train6 = np.asarray(y_train6)

f = open("../priorWorks/TestData/relocs_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_test6 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test6 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test6.append(indexOfLabel)
    count += 1
y_test6 = np.asarray(y_test6)


print(x_train6.shape)
print(y_train6.shape)
print(x_test6.shape)
print(y_test6.shape)





f = open("../priorWorks/TrainData/sections_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_train7 = np.asarray(data)
families = np.asarray(families)


y_train7 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train7.append(indexOfLabel)
    count += 1
y_train7 = np.asarray(y_train7)

f = open("../priorWorks/TestData/sections_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_test7 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test7 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test7.append(indexOfLabel)
    count += 1
y_test7 = np.asarray(y_test7)


print(x_train7.shape)
print(y_train7.shape)
print(x_test7.shape)
print(y_test7.shape)



f = open("../priorWorks/TrainData/segments_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_train8 = np.asarray(data)
families = np.asarray(families)


y_train8 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train8.append(indexOfLabel)
    count += 1
y_train8 = np.asarray(y_train8)

f = open("../priorWorks/TestData/segments_sizes-FamOrder.pickle","rb")
data,families = pickle.load(f)
x_test8 = np.asarray(data)

f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test8 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test8.append(indexOfLabel)
    count += 1
y_test8 = np.asarray(y_test8)


print(x_train8.shape)
print(y_train8.shape)
print(x_test8.shape)
print(y_test8.shape)




f = open("../priorWorks/TrainData/trainData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
data = data.todense()

x_train9 = np.asarray(data)
families = np.asarray(families)

# print(x_train9.shape)
# exit()

y_train9 = []
count = 0
for i in range(len(data)):
    indexOfLabel = FO.index(families[int(count/100)])
    y_train9.append(indexOfLabel)
    count += 1
y_train9 = np.asarray(y_train9)

f = open("../priorWorks/TestData/testData-FamOrder-Edited.pickle","rb")
data,families = pickle.load(f)
data = data.todense()
x_test9 = np.asarray(data)
f = open("../priorWorks/testFamily_MalwareList.pickle","rb")
countRaw = pickle.load(f)
countTotal = []
for i in range(len(families)):
    countTotal.append(len(countRaw[families[i]]))

y_test9 = []
count = 0
for i in range(len(data)):
    rawCount = 0
    index = 0
    for j in range(len(countTotal)):
        if count >= rawCount and count < (rawCount+countTotal[j]):
            index = j
            break
        rawCount = rawCount+countTotal[j]
    indexOfLabel = FO.index(families[index])
    y_test9.append(indexOfLabel)
    count += 1
y_test9 = np.asarray(y_test9)


print(x_train9.shape)
print(y_train9.shape)
print(x_test9.shape)
print(y_test9.shape)

x_train = np.concatenate((x_train1, x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,x_train9), axis=1)
x_test = np.concatenate((x_test1, x_test2,x_test3,x_test4,x_test5,x_test6,x_test7,x_test8,x_test9), axis=1)
y_train = y_train1
y_test = y_test1

print(x_train.shape)
print(x_test.shape)

# print("==========================================================================")
# print("LR")
# print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# print("The LR score is:",score)
#
# print("==========================================================================")
# print("RF")
# print("==========================================================================")
# clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# print("The RF score is:",score)
#
# print("==========================================================================")
# print("KNN")
# print("==========================================================================")
# neighbors = [3,5,10]
# for neighbor in neighbors:
#     clf = KNeighborsClassifier(n_neighbors=neighbor).fit(x_train, y_train)
#     score = clf.score(x_test, y_test)
#     print(neighbor,"The KNN score is:",score)
# from sklearn.svm import SVC
# print("==========================================================================")
# print("SVM")
# print("==========================================================================")
# clf = SVC().fit(x_train, y_train)
# score = clf.score(x_test, y_test)
# print("The SVC score is:",score)


# print("==========================================================================")
# print("Deep Learning")
# print("==========================================================================")
# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# # create model
# model = Sequential()
# model.add(keras.layers.Dense(16, activation='relu',input_shape=(x_train.shape[1:])))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Dense(32, activation='relu'))
# # model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# # model.add(keras.layers.Dense(256, activation='relu'))
# # model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(10, activation="softmax"))
#
# # Compile model
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=10, batch_size=32)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test accuracy:', scores[1])



print("==========================================================================")
print("CNN")
print("==========================================================================")
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# create model
model = Sequential()

filter = 16
# create model
model = Sequential()
model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
# model.add(keras.layers.MaxPooling1D())
# model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Conv1D(filter*3,3,padding="same",activation="relu"))
# model.add(keras.layers.Conv1D(filter*3,3,padding="valid",activation="relu"))
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
