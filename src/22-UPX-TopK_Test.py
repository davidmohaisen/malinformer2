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
import pickle5 as pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score



families = ['mirai', 'xorddos', 'ganiw', 'tsunami', 'gafgyt', 'elknot', 'generica', 'setag', 'dofloo', 'local']
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
for i in np.arange(500,20500,500):
    x_train, x_test, y_train, y_test = [], [], [], []

    for family in families:
        f = open("/media/ahmed/HDD/FamilyAnalysis/topK_dynamic/transformedOnTopKFeats/"+family+"-"+str(i)+"-frequency-Train.pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)

        if len(x_train) == 0:
            x_train = partial_x
            y_train = y_partial

        else:
            x_train = np.concatenate((x_train,partial_x))
            y_train = np.concatenate((y_train,y_partial))

        f = open("/media/ahmed/HDD/FamilyAnalysis/topK_dynamic/transformedOnTopKFeats/"+family+"-"+str(i)+"-frequency-Test.pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)
        if len(x_test) == 0:
            x_test = partial_x
            y_test = y_partial
        else:
            x_test = np.concatenate((x_test,partial_x))
            y_test = np.concatenate((y_test,y_partial))

    print(x_train.shape)
    print(x_test.shape)


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

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The RF score is:",score)
    RFResults.append(score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    F1RFResults.append(f1_score(y_true, y_pred, average='weighted'))


    print("==========================================================================")
    print("KNN")
    print("==========================================================================")
    countKNN = 0
    neighbors = [3,5,10]
    for neighbor in neighbors:
        clf = KNeighborsClassifier(n_neighbors=neighbor).fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print(neighbor,"The KNN score is:",score)
        KNNResults[countKNN].append(score)
        y_true = y_test
        y_pred = clf.predict(x_test)
        F1KNNResults[countKNN].append(f1_score(y_true, y_pred, average='weighted'))

        countKNN += 1

    from sklearn.svm import SVC
    print("==========================================================================")
    print("SVM")
    print("==========================================================================")
    clf = SVC().fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The SVC score is:",score)
    SVMResults.append(score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    F1SVMResults.append(f1_score(y_true, y_pred, average='weighted'))

    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
    # create model
    try:
        model = Sequential()
        model.add(keras.layers.Dense(16, activation='relu',input_shape=(x_train.shape[1:])))
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(128, activation='relu'))
        # model.add(keras.layers.Dropout(0.25))
        # model.add(keras.layers.Flatten())
        # model.add(keras.layers.Dense(256, activation='relu'))
        # model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation="softmax"))

        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        model.fit(x_train,y_train, epochs=10, batch_size=16)
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', scores[1])
        DNNResults.append(scores[1])
        y_true = y_test
        y_pred = np.argmax(model.predict(x_test),axis=1)
        F1DNNResults.append(f1_score(y_true, y_pred, average='weighted'))
    except:
        DNNResults.append("-")
        F1DNNResults.append("-")
    #
    #
    print("==========================================================================")
    print("CNN")
    print("==========================================================================")
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    # create model

    filter = 4
    # create model
    try:
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
            model.fit(x_train,y_train, epochs=10, batch_size=32)
        except:
            try:
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
            except:
                filter = 1
                model = Sequential()
                model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
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
            model.fit(x_train,y_train, epochs=10, batch_size=8)
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy:', scores[1])
        CNNResults.append(scores[1])
        y_true = y_test
        y_pred = np.argmax(model.predict(x_test),axis=1)
        F1CNNResults.append(f1_score(y_true, y_pred, average='weighted'))
    except:
        CNNResults.append("-")
        F1CNNResults.append("-")

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
print("DNN")

print(DNNResults)
print(F1DNNResults)
print("CNN")

print(CNNResults)
print(F1CNNResults)
