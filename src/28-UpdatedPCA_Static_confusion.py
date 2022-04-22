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
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

variance = ["0.999"]

for var in variance:
    print(var)
    print("_________________________________________________")
    f = open("../PCADynamicStatic/static/PCA2components-Train-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_train = np.asarray(data)


    y_train = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_train.append(indexOfLabel)
    y_train = np.asarray(y_train)

    f = open("../PCADynamicStatic/static/PCA2components-Test-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_test = np.asarray(data)

    y_test = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_test.append(indexOfLabel)
    y_test = np.asarray(y_test)


    y_test = np.asarray(y_test)



    f = open("../PCADynamicStatic/static/PCA2components-Upx-family_variance-"+var+".pickle","rb")
    data,families = pickle.load(f)
    x_packed = np.asarray(data)

    y_packed = []
    for i in range(len(families)):
        indexOfLabel = FO.index(families[i])
        y_packed.append(indexOfLabel)
    y_packed = np.asarray(y_packed)

    f = open("../PCADynamicStatic/static/PCA2components-UpxUnpacked-family_variance-"+var+".pickle","rb")
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
    print("RF")
    print("==========================================================================")
    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("The rf score is:",score)
    y_true = y_test
    y_pred = clf.predict(x_test)
    mat = confusion_matrix(y_true, y_pred)
    matN = []
    for i in range(len(mat)):
        matN.append(list(1.0*mat[i]/sum(mat[i])))
    for i in range(len(matN)):
        for j in range(len(matN[i])):
            # matN[i][j] = round(matN[i][j],3)
            matN[i][j] = round(matN[i][j],2)
    matN = np.asarray(matN)
    print(matN)
    df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
    # sns.set(font_scale=0.85)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
    plt.savefig("../Heatmaps/HM-StaticPCATest_2p.pdf")
    plt.clf()
    plt.close()


    score = clf.score(x_unpacked, y_unpacked)
    print("The rf score is:",score)
    y_true = y_unpacked
    y_pred = clf.predict(x_unpacked)
    mat = confusion_matrix(y_true, y_pred)
    matN = []
    for i in range(len(mat)):
        matN.append(list(1.0*mat[i]/sum(mat[i])))
    for i in range(len(matN)):
        for j in range(len(matN[i])):
            # matN[i][j] = round(matN[i][j],3)
            matN[i][j] = round(matN[i][j],2)
    matN = np.asarray(matN)
    print(matN)
    df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
    # sns.set(font_scale=0.85)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
    plt.savefig("../Heatmaps/HM-StaticPCAUnpacked_2p.pdf")
    plt.clf()
    plt.close()

    score = clf.score(x_packed, y_packed)
    print("The rf score is:",score)
    y_true = y_packed
    y_pred = clf.predict(x_packed)
    mat = confusion_matrix(y_true, y_pred)
    matN = []
    for i in range(len(mat)):
        matN.append(list(1.0*mat[i]/sum(mat[i])))
    for i in range(len(matN)):
        for j in range(len(matN[i])):
            # matN[i][j] = round(matN[i][j],3)
            matN[i][j] = round(matN[i][j],2)
    matN = np.asarray(matN)
    print(matN)
    df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
    # sns.set(font_scale=0.85)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
    plt.savefig("../Heatmaps/HM-StaticPCAPacked_2p.pdf")
    plt.clf()
    plt.close()
