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
from sklearn.metrics import f1_score

os.environ['KMP_DUPLICATE_LIB_OK']='True'
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

variance = ["0.85","0.9","0.95","0.99","0.999"]
LRResults = []
RFResults = []
KNNResults = [[],[],[]]
SVMResults = []
DNNResults = []
CNNResults = []

F1LRResults = []
F1RFResults = []
F1KNNResults = [[],[],[]]
F1SVMResults = []
F1DNNResults = []
F1CNNResults = []



LRResults_unpacked = []
RFResults_unpacked = []
KNNResults_unpacked = [[],[],[]]
SVMResults_unpacked = []
DNNResults_unpacked = []
CNNResults_unpacked = []

F1LRResults_unpacked = []
F1RFResults_unpacked = []
F1KNNResults_unpacked = [[],[],[]]
F1SVMResults_unpacked = []
F1DNNResults_unpacked = []
F1CNNResults_unpacked = []


LRResults_packed = []
RFResults_packed = []
KNNResults_packed = [[],[],[]]
SVMResults_packed = []
DNNResults_packed = []
CNNResults_packed = []

F1LRResults_packed = []
F1RFResults_packed = []
F1KNNResults_packed = [[],[],[]]
F1SVMResults_packed = []
F1DNNResults_packed = []
F1CNNResults_packed = []

for var in variance:
    print(var)
    print("_________________________________________________")
    f = open("../PCADynamicStatic/dynamic/PCA2components-Train-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_train = np.asarray(data)


    y_train = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_train.append(indexOfLabel)
    y_train = np.asarray(y_train)

    f = open("../PCADynamicStatic/dynamic/PCA2components-Test-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_test = np.asarray(data)

    y_test = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_test.append(indexOfLabel)
    y_test = np.asarray(y_test)


    y_test = np.asarray(y_test)



    f = open("../PCADynamicStatic/dynamic/PCA2components-Upx-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_packed = np.asarray(data)

    y_packed = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_packed.append(indexOfLabel)
    y_packed = np.asarray(y_packed)

    f = open("../PCADynamicStatic/dynamic/PCA2components-UpxUnpacked-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_unpacked = np.asarray(data)

    y_unpacked = y_packed

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_unpacked.shape)
    print(y_unpacked.shape)
    print(x_packed.shape)
    print(y_packed.shape)

    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The LR score is:",score)
    LRResults.append(score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    F1LRResults.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_unpacked, y_unpacked)
    print("The LR score is:",score)
    LRResults_unpacked.append(score)
    y_true = y_unpacked
    y_pred = clf.predict(x_unpacked)
    F1LRResults_unpacked.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_packed, y_packed)
    print("The LR score is:",score)
    LRResults_packed.append(score)
    y_true = y_packed
    y_pred = clf.predict(x_packed)
    F1LRResults_packed.append(f1_score(y_true, y_pred, average='weighted'))

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The LR score is:",score)
    RFResults.append(score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    F1RFResults.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_unpacked, y_unpacked)
    print("The LR score is:",score)
    RFResults_unpacked.append(score)
    y_true = y_unpacked
    y_pred = clf.predict(x_unpacked)
    F1RFResults_unpacked.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_packed, y_packed)
    print("The LR score is:",score)
    RFResults_packed.append(score)
    y_true = y_packed
    y_pred = clf.predict(x_packed)
    F1RFResults_packed.append(f1_score(y_true, y_pred, average='weighted'))

    print("==========================================================================")
    print("KNN")
    print("==========================================================================")
    countKNN = 0
    neighbors = [3,5,10]
    for neighbor in neighbors:
        clf = KNeighborsClassifier(n_neighbors=neighbor).fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print("The LR score is:",score)
        KNNResults[countKNN].append(score)
        y_true = y_test
        y_pred = clf.predict(x_test)
        F1KNNResults[countKNN].append(f1_score(y_true, y_pred, average='weighted'))

        score = clf.score(x_unpacked, y_unpacked)
        print("The LR score is:",score)
        KNNResults_unpacked[countKNN].append(score)
        y_true = y_unpacked
        y_pred = clf.predict(x_unpacked)
        F1KNNResults_unpacked[countKNN].append(f1_score(y_true, y_pred, average='weighted'))

        score = clf.score(x_packed, y_packed)
        print("The LR score is:",score)
        KNNResults_packed[countKNN].append(score)
        y_true = y_packed
        y_pred = clf.predict(x_packed)
        F1KNNResults_packed[countKNN].append(f1_score(y_true, y_pred, average='weighted'))

        countKNN += 1

    from sklearn.svm import SVC
    print("==========================================================================")
    print("SVM")
    print("==========================================================================")
    clf = SVC().fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The LR score is:",score)
    SVMResults.append(score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    F1SVMResults.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_unpacked, y_unpacked)
    print("The LR score is:",score)
    SVMResults_unpacked.append(score)
    y_true = y_unpacked
    y_pred = clf.predict(x_unpacked)
    F1SVMResults_unpacked.append(f1_score(y_true, y_pred, average='weighted'))

    score = clf.score(x_packed, y_packed)
    print("The LR score is:",score)
    SVMResults_packed.append(score)
    y_true = y_packed
    y_pred = clf.predict(x_packed)
    F1SVMResults_packed.append(f1_score(y_true, y_pred, average='weighted'))



    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    x_unpacked = np.reshape(x_unpacked,(x_unpacked.shape[0],x_unpacked.shape[1],1))
    x_packed = np.reshape(x_packed,(x_packed.shape[0],x_packed.shape[1],1))
    # create model
    model = Sequential()
    model.add(keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1:])))
    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=20, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    DNNResults.append(scores[1])
    y_true = y_test
    y_pred = np.argmax(model.predict(x_test),axis=1)
    F1DNNResults.append(f1_score(y_true, y_pred, average='weighted'))

    scores = model.evaluate(x_unpacked, y_unpacked, verbose=1)
    print('Test accuracy:', scores[1])
    DNNResults_unpacked.append(scores[1])
    y_true = y_unpacked
    y_pred = np.argmax(model.predict(x_unpacked),axis=1)
    F1DNNResults_unpacked.append(f1_score(y_true, y_pred, average='weighted'))

    scores = model.evaluate(x_packed, y_packed, verbose=1)
    print('Test accuracy:', scores[1])
    DNNResults_packed.append(scores[1])
    y_true = y_packed
    y_pred = np.argmax(model.predict(x_packed),axis=1)
    F1DNNResults_packed.append(f1_score(y_true, y_pred, average='weighted'))

    #
    #
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
    try:
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
        model.fit(x_train,y_train, epochs=100, batch_size=32)
    except:
        model = Sequential()
        model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
        model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
        # model.add(keras.layers.MaxPooling1D())
        # model.add(keras.layers.Dropout(0.25))
        # model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
        # model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
        # # model.add(keras.layers.MaxPooling1D())
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
        model.fit(x_train,y_train, epochs=20, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    CNNResults.append(scores[1])
    y_true = y_test
    y_pred = np.argmax(model.predict(x_test),axis=1)
    F1CNNResults.append(f1_score(y_true, y_pred, average='weighted'))

    scores = model.evaluate(x_unpacked, y_unpacked, verbose=1)
    print('Test accuracy:', scores[1])
    CNNResults_unpacked.append(scores[1])
    y_true = y_unpacked
    y_pred = np.argmax(model.predict(x_unpacked),axis=1)
    F1CNNResults_unpacked.append(f1_score(y_true, y_pred, average='weighted'))

    scores = model.evaluate(x_packed, y_packed, verbose=1)
    print('Test accuracy:', scores[1])
    CNNResults_packed.append(scores[1])
    y_true = y_packed
    y_pred = np.argmax(model.predict(x_packed),axis=1)
    F1CNNResults_packed.append(f1_score(y_true, y_pred, average='weighted'))

print("Test")
print("LR")
print(LRResults)
print(F1LRResults)
print("RF")
print(RFResults)
print(F1RFResults)
print("KNN 3")
print(KNNResults[0])
print(F1KNNResults[0])
print("KNN 5")
print(KNNResults[1])
print(F1KNNResults[1])
print("KNN 10")
print(KNNResults[2])
print(F1KNNResults[2])
print("SVM")
print(SVMResults)
print(F1SVMResults)
print("DNN)")
print(DNNResults)
print(F1DNNResults)
print("CNN")
print(CNNResults)
print(F1CNNResults)


print("unpacked")
print("LR")
print(LRResults_unpacked)
print(F1LRResults_unpacked)
print("RF")
print(RFResults_unpacked)
print(F1RFResults_unpacked)
print("KNN 3")
print(KNNResults_unpacked[0])
print(F1KNNResults_unpacked[0])
print("KNN 5")
print(KNNResults_unpacked[1])
print(F1KNNResults_unpacked[1])
print("KNN 10")
print(KNNResults_unpacked[2])
print(F1KNNResults_unpacked[2])
print("SVM")
print(SVMResults_unpacked)
print(F1SVMResults_unpacked)
print("DNN)")
print(DNNResults_unpacked)
print(F1DNNResults_unpacked)
print("CNN")
print(CNNResults_unpacked)
print(F1CNNResults_unpacked)


print("packed")
print("LR")
print(LRResults_packed)
print(F1LRResults_packed)
print("RF")
print(RFResults_packed)
print(F1RFResults_packed)
print("KNN 3")
print(KNNResults_packed[0])
print(F1KNNResults_packed[0])
print("KNN 5")
print(KNNResults_packed[1])
print(F1KNNResults_packed[1])
print("KNN 10")
print(KNNResults_packed[2])
print(F1KNNResults_packed[2])
print("SVM")
print(SVMResults_packed)
print(F1SVMResults_packed)
print("DNN)")
print(DNNResults_packed)
print(F1DNNResults_packed)
print("CNN")
print(CNNResults_packed)
print(F1CNNResults_packed)
