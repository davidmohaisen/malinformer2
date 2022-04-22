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
from sklearn.neighbors import KNeighborsClassifier
families = ['mirai', 'xorddos', 'ganiw', 'tsunami', 'gafgyt', 'elknot', 'generica', 'setag', 'dofloo', 'local']
ctr_files = {}
# families = set()
for file in os.listdir('../featureAnalysis/discriminativeFeatsOccurrences/'):
    num = int(file.rsplit("-")[2])
    # families.add(file.rsplit("-")[1])
    if num not in ctr_files:
        ctr_files[num] = set()
        ctr_files[num].add('../featureAnalysis/discriminativeFeatsOccurrences/'+file)
    else:
        ctr_files[num].add('../featureAnalysis/discriminativeFeatsOccurrences/'+file)

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
            y.append(families.index(file.rsplit("-")[1]))
    # x = np.asarray(x)
    # y = np.asarray(y)
    # print(x.shape)
    # print(y.shape)
    # exit()
    x =  np.array(x)
    # print(x.shape)
    x_train, x_test, y_train, y_test = [],[],[],[]

    total = 100
    subcurrent = []
    for i in range(len(families)):
        subcurrent.append(0)
    for i in range(len(x)):
        if subcurrent[y[i]] < 0.8*total:
            subcurrent[y[i]]+= 1
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            subcurrent[y[i]]+= 1
            x_test.append(x[i])
            y_test.append(y[i])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    if num != 9500:
        continue
    for NN in range(1,51):
        neigh = KNeighborsClassifier(n_neighbors=NN,n_jobs=-1)
        neigh.fit(x_train, y_train)
        score = neigh.score(x_test, y_test)
        print("Model",num, "NN", NN)
        print(score)
    continue
    y_test = keras.utils.to_categorical(y_test, len(families))
    y_train = keras.utils.to_categorical(y_train, len(families))

    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    filter = 64
    # create model
    model = Sequential()
    model.add(keras.layers.Dense(100, activation="relu",input_shape=(x_train.shape[1:])))
    # model.add(keras.layers.Dense(100, activation="relu",input_shape=(x_train.shape[1:])))
    # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.Dense(100, activation="relu",input_shape=(x_train.shape[1:])))
    model.add(keras.layers.Dense(100, activation="relu",input_shape=(x_train.shape[1:])))

    # model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu",input_shape=(x_train.shape[1:])))
    # model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
    # model.add(keras.layers.MaxPooling1D())
    # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
    # model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))

    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(families), activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    # Fit the model\
    model.fit(x_train,y_train, epochs=10, batch_size=16)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Model",num)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
