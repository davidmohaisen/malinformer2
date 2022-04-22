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
families = ['gafgyt', 'mirai', 'xorddos', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

for i in np.arange(4000,4500,500):
    x_train, x_test, y_train, y_test = [], [], [], []

    for family in families:
        f = open("../topK_dynamic/transformedOnTop20K_ExceptTsunamiGenericaFeats/"+family+"-"+str(i)+"-frequency-Train.pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)

        if len(x_train) == 0:
            x_train = partial_x
            y_train = y_partial

        else:
            x_train = np.concatenate((x_train,partial_x))
            y_train = np.concatenate((y_train,y_partial))

        f = open("../topK_dynamic/transformedOnTop20K_ExceptTsunamiGenericaFeats/"+family+"-"+str(i)+"-frequency-Packed.pickle","rb")
        partial_x = pickle.load(f)
        partial_x = np.asarray(partial_x)
        y_partial = [families.index(family)]*len(partial_x)
        if len(x_test) == 0:
            x_test = partial_x
            y_test = y_partial
        else:
            x_test = np.concatenate((x_test,partial_x))
            y_test = np.concatenate((y_test,y_partial))
    print(len(x_test))
    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The RF score is:",score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    print(f1_score(y_true, y_pred, average='weighted'))

    mat = confusion_matrix(y_true, y_pred)
    matN = []
    for ik in range(len(mat)):
        matN.append(list(1.0*mat[ik]/sum(mat[ik])))
    for ik in range(len(matN)):
        for j in range(len(matN[ik])):
            matN[ik][j] = round(matN[ik][j],3)
    matN = np.asarray(matN)
    print(matN)
    df = pd.DataFrame(matN, columns=["F0","F1","F2","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F5","F6","F7","F8","F9"])
    sns.set(font_scale=0.85)
    ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
    plt.savefig("../Heatmaps/HM-DynamicTopKPacked_except.pdf")
    plt.clf()
    plt.close()
