
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
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
families = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

fout = open("../data/dynamic/retrainOutput.csv","w")
fout.write("Instructions,Test Accuracy, Test F-1, Unpacked Accuracy, Unpacked F-1, Packed Accuracy, Packed F-1,\n")
fout2 = open("../data/dynamic/pretrainOutput.csv","w")
fout2.write("Instructions,Test Accuracy, Test F-1, Unpacked Accuracy, Unpacked F-1, Packed Accuracy, Packed F-1,\n")

f = open("../data/dynamic/transformedOnTop20K_InstrCount/RFBaselineModel.pickle","rb")
preCLF = pickle.load(f)
N=250000
for t in range(0,100):
    i = int((t+1)*(N/100))
    fout.write(str(i)+",")
    fout2.write(str(i)+",")
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_unpacked, y_unpacked = [], []
    x_packed, y_packed = [], []

    for family in families:
        f = open("../data/dynamic/transformedOnTop20K_InstrCount/"+family+"-4000-frequency-Train-InstrCntPerc"+str(i)+".pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)

        if len(x_train) == 0:
            x_train = partial_x
            y_train = y_partial

        else:
            x_train = np.concatenate((x_train,partial_x))
            y_train = np.concatenate((y_train,y_partial))

        f = open("../data/dynamic/transformedOnTop20K_InstrCount/"+family+"-4000-frequency-Test-InstrCntPerc"+str(i)+".pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)
        if len(x_test) == 0:
            x_test = partial_x
            y_test = y_partial
        else:
            x_test = np.concatenate((x_test,partial_x))
            y_test = np.concatenate((y_test,y_partial))

        f = open("../data/dynamic/transformedOnTop20K_InstrCount/"+family+"-4000-frequency-Unpacked-InstrCntPerc"+str(i)+".pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)
        if len(x_unpacked) == 0:
            x_unpacked = partial_x
            y_unpacked = y_partial
        else:
            x_unpacked = np.concatenate((x_unpacked,partial_x))
            y_unpacked = np.concatenate((y_unpacked,y_partial))
        f = open("../data/dynamic/transformedOnTop20K_InstrCount/"+family+"-4000-frequency-Packed-InstrCntPerc"+str(i)+".pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)
        if len(x_packed) == 0:
            x_packed = partial_x
            y_packed = y_partial
        else:
            x_packed = np.concatenate((x_packed,partial_x))
            y_packed = np.concatenate((y_packed,y_partial))
    print(len(x_test))


    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    fout.write(str(score)+",")
    y_true = y_test
    y_pred = clf.predict(x_test)
    fout.write(str(f1_score(y_true, y_pred, average='weighted'))+",")


    score = clf.score(x_unpacked, y_unpacked)
    fout.write(str(score)+",")
    y_true = y_unpacked
    y_pred = clf.predict(x_unpacked)
    fout.write(str(f1_score(y_true, y_pred, average='weighted'))+",")

    score = clf.score(x_packed, y_packed)
    fout.write(str(score)+",")
    y_true = y_packed
    y_pred = clf.predict(x_packed)
    fout.write(str(f1_score(y_true, y_pred, average='weighted'))+",\n")



    score = preCLF.score(x_test, y_test)
    fout2.write(str(score)+",")
    y_true = y_test
    y_pred = preCLF.predict(x_test)
    fout2.write(str(f1_score(y_true, y_pred, average='weighted'))+",")


    score = preCLF.score(x_unpacked, y_unpacked)
    fout2.write(str(score)+",")
    y_true = y_unpacked
    y_pred = preCLF.predict(x_unpacked)
    fout2.write(str(f1_score(y_true, y_pred, average='weighted'))+",")

    score = preCLF.score(x_packed, y_packed)
    fout2.write(str(score)+",")
    y_true = y_packed
    y_pred = preCLF.predict(x_packed)
    fout2.write(str(f1_score(y_true, y_pred, average='weighted'))+",\n")
