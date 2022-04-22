from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from os import listdir
from os.path import isfile, join
import collections
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
families = ['mirai', 'xorddos', 'ganiw', 'tsunami', 'gafgyt', 'elknot', 'generica', 'setag', 'dofloo', 'local']
ctr_files = {}
# families = set()
# for file in os.listdir('../noThreshold/'):
for file in os.listdir('../noThresholdWithDuplicatedIndices/'):
    num = int(file.rsplit("-")[1])
    # families.add(file.rsplit("-")[1])
    if num not in ctr_files:
        ctr_files[num] = set()
        # ctr_files[num].add('../noThreshold/'+file)
        ctr_files[num].add('../noThresholdWithDuplicatedIndices/'+file)
    else:
        # ctr_files[num].add('../noThreshold/'+file)
        ctr_files[num].add('../noThresholdWithDuplicatedIndices/'+file)

for num in ctr_files:
    # modules = range(num-501,num+1,50)
    # for module in modules:
    x = []
    y = []
    # path = "../featureAnalysis/discriminativeFeatsOccurrences/"
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for file in ctr_files[num]:
        f = open(file,"rb")
        Data = pickle.load(f)
        # print(file,Data.shape)
        for j in range(len(Data)):
            x.append(Data[j])
            # print(file)
            # file = file.rsplit("/")[2]
            # print(file[2])
            y.append(families.index(file.rsplit("/")[2].rsplit("-")[0]))
    x = np.asarray(x)
    y = np.asarray(y)
    # print(num,x.shape)
    # print(num,y.shape)
    # exit()
    x_train, x_test, y_train, y_test = [], [], [], []
    total_y = collections.Counter(y)
    # print(total_y)
    current_count = [0]*len(families)

    for i in range(len(x)):
        # split = 0.8*100
        if current_count[y[i]] >= 0.8*total_y[y[i]]:
            x_test.append(x[i])
            y_test.append(y[i])
            current_count[y[i]] += 1
        else:
            current_count[y[i]] += 1
            x_train.append(x[i])
            y_train.append(y[i])
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    x =  np.array(x)
    # create model
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    y_test = keras.utils.to_categorical(y_test, len(families))
    y_train = keras.utils.to_categorical(y_train, len(families))
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # print(y_train.shape)
    filter = 32

    model = Sequential()
    model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu"))
    model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
    model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
    model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
    model.add(keras.layers.MaxPooling1D())
    # model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'],verbose=0)
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    # Fit the model\
    model.fit(x_train,y_train, epochs=20, batch_size=32)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(num, scores[1])

    preds = model.predict(x_test)
    con = confusion_matrix(y_test, preds)
    print(con)
