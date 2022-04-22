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
for var in variance:
    print(var)
    print("_________________________________________________")
    f = open("../PCADynamicAnalysis/PCA2components-Train-family_variance-"+var+".pickle","rb")
    data,y_label = pickle.load(f)
    exit()
    f = open("../PCADynamicAnalysis/Train_Family.pickle","rb")
    families = pickle.load(f)

    x_train = np.asarray(data)


    y_train = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_train.append(indexOfLabel)
    y_train = np.asarray(y_train)


    f = open("../PCADynamicAnalysis/PCA2components-Test-family_variance-"+var+".pickle","rb")
    data = pickle.load(f)
    x_test = np.asarray(data)

    f = open("../PCADynamicAnalysis/Test_Family.pickle","rb")
    families = pickle.load(f)


    y_test = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_test.append(indexOfLabel)
    y_test = np.asarray(y_test)


    y_test = np.asarray(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


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
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
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
    model.fit(x_train,y_train, epochs=10, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    DNNResults.append(scores[1])
    y_true = y_test
    y_pred = np.argmax(model.predict(x_test),axis=1)
    F1DNNResults.append(f1_score(y_true, y_pred, average='weighted'))

    #
    #
    print("==========================================================================")
    print("CNN")
    print("==========================================================================")
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
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
        model.fit(x_train,y_train, epochs=100, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    CNNResults.append(scores[1])
    y_true = y_test
    y_pred = np.argmax(model.predict(x_test),axis=1)
    F1CNNResults.append(f1_score(y_true, y_pred, average='weighted'))

print(LRResults)
print(F1LRResults)
print(RFResults)
print(F1RFResults)
print(KNNResults[0])
print(F1KNNResults[0])
print(KNNResults[1])
print(F1KNNResults[1])
print(KNNResults[2])
print(F1KNNResults[2])
print(SVMResults)
print(F1SVMResults)
print(DNNResults)
print(F1DNNResults)
print(CNNResults)
print(F1CNNResults)
