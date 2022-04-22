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
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

representationLower = "entropy"
representationUpper = "Entropy"


pathTrain = "../priorWorks/updatedData/trainingMalData/"+representationLower+"/"
pathTest = "../priorWorks/updatedData/testDataset/"+representationLower+"/"

f = open("../priorWorks/updatedData/trainFamily_MalwareList.pickle","rb")
labels = pickle.load(f)





x_train = []
y_train = []
x_test = []
y_test = []

for fo_index in range(len(FO)):
    for file_index in range(len(labels[FO[fo_index]])):
        FileToDealWith = labels[FO[fo_index]][file_index]

        try:
            f = open(pathTrain+FileToDealWith+".pickle","rb")
            x = pickle.load(f)
            x = x[:10000]
            while len(x)!= 10000:
                x.append(0)
            x_train.append(x)
            y_train.append(fo_index)
        except:
            try:
                f = open(pathTrain+"VirusShare_"+FileToDealWith+".pickle","rb")
                x = pickle.load(f)
                x = x[:10000]
                while len(x)!= 10000:
                    x.append(0)
                x_train.append(x)
                y_train.append(fo_index)
            except:
                # x = []
                # while len(x)!= 10000:
                #     x.append(0)
                # x_train.append(x)
                # y_train.append(fo_index)
                continue

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print(x_train.shape)
print(y_train.shape)

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1", "wb")
pickle.dump([x_train[:int(len(x_train)/2)],y_train[:int(len(x_train)/2)]],f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2", "wb")
pickle.dump([x_train[int(len(x_train)/2):],y_train[int(len(x_train)/2):]],f)




del x_train,y_train


f = open("../priorWorks/updatedData/testFamily_MalwareList.pickle","rb")
labels_test = pickle.load(f)


for fo_index in range(len(FO)):
    for file_index in range(len(labels_test[FO[fo_index]])):
        FileToDealWith = labels_test[FO[fo_index]][file_index]

        try:
            f = open(pathTest+FileToDealWith+".pickle","rb")
            x = pickle.load(f)
            x = x[:10000]
            while len(x)!= 10000:
                x.append(0)
            x_test.append(x)
            y_test.append(fo_index)
        except:
            try:
                f = open(pathTest+"VirusShare_"+FileToDealWith+".pickle","rb")
                x = pickle.load(f)
                x = x[:10000]
                while len(x)!= 10000:
                    x.append(0)
                x_test.append(x)
                y_test.append(fo_index)
            except FileNotFoundError:
                continue



x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData", "wb")
pickle.dump([x_test,y_test],f)
print(x_test.shape)
print(y_test.shape)

del x_test,y_test


f = open("../priorWorks/updatedData/Packed-fam_malw_MalwareList.pickle","rb")
labels_packed = pickle.load(f)

x_unpacked = []
y_unpacked = []
for fo_index in range(len(FO)):
    for file_index in range(len(labels_packed[FO[fo_index]])):
        FileToDealWith = labels_packed[FO[fo_index]][file_index]

        try:
            f = open(pathTest+FileToDealWith+".pickle","rb")
            x = pickle.load(f)
            x = x[:10000]
            while len(x)!= 10000:
                x.append(0)
            x_unpacked.append(x)
            y_unpacked.append(fo_index)
        except:
            try:
                f = open(pathTest+"VirusShare_"+FileToDealWith+".pickle","rb")
                x = pickle.load(f)
                x = x[:10000]
                while len(x)!= 10000:
                    x.append(0)
                x_unpacked.append(x)
                y_unpacked.append(fo_index)
            except:
                continue


x_unpacked = np.asarray(x_unpacked)
y_unpacked = np.asarray(y_unpacked)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData", "wb")
pickle.dump([x_unpacked,y_unpacked],f)

print(x_unpacked.shape)
print(y_unpacked.shape)

del x_unpacked,y_unpacked


pathPacked = "../priorWorks/updatedData/upxPacked/"+representationLower+"/"
x_packed = []
y_packed = []
for fo_index in range(len(FO)):
    for file_index in range(len(labels_packed[FO[fo_index]])):
        FileToDealWith = labels_packed[FO[fo_index]][file_index]

        try:
            f = open(pathPacked+FileToDealWith+".pickle","rb")
            x = pickle.load(f)
            x = x[:10000]
            while len(x)!= 10000:
                x.append(0)
            x_packed.append(x)
            y_packed.append(fo_index)
        except:
            try:
                f = open(pathPacked+"VirusShare_"+FileToDealWith+".pickle","rb")
                x = pickle.load(f)
                x = x[:10000]
                while len(x)!= 10000:
                    x.append(0)
                x_packed.append(x)
                y_packed.append(fo_index)
            except:
                continue




x_packed = np.asarray(x_packed)
y_packed = np.asarray(y_packed)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData", "wb")
pickle.dump([x_packed,y_packed],f)

print(x_packed.shape)
print(y_packed.shape)

del x_packed,y_packed



# exit()

f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train = np.concatenate((x_train_1,x_train_2))
y_train = np.concatenate((y_train_1,y_train_2))
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"TestData","rb")
x_test, y_test = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"UnpackedData","rb")
x_unpacked, y_unpacked = pickle.load(f)
f = open("../priorWorks/updatedData/pickles/"+representationUpper+"PackedData","rb")
x_packed, y_packed = pickle.load(f)
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
print("The RF score is:",score)
y_true = y_test
y_pred = clf.predict(x_test)
print(f1_score(y_true, y_pred, average='weighted'))
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
plt.savefig("../Heatmaps/HM-StaticEntropyTest_2p.pdf")
plt.clf()
plt.close()

score = clf.score(x_unpacked, y_unpacked)
print("The RF score unpacked is:",score)
y_true = y_unpacked
y_pred = clf.predict(x_unpacked)
print(f1_score(y_true, y_pred, average='weighted'))
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
plt.savefig("../Heatmaps/HM-StaticEntropyUnpacked_2p.pdf")
plt.clf()
plt.close()

score = clf.score(x_packed, y_packed)
print("The RF score packed is:",score)
y_true = y_packed
y_pred = clf.predict(x_packed)
print(f1_score(y_true, y_pred, average='weighted'))

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
# plt.rcParams['font.size'] = 20
df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
# sns.set(font_scale=0.85)
sns.set(font_scale=1.2)
ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')

# ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.75,vmin=-0.02,vmax=1.0)
plt.savefig("../Heatmaps/HM-StaticEntropyPacked_2p.pdf")
plt.clf()
plt.close()
