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

x_test_unpacked, y_test_unpacked = [], []
x_train, y_train = [], []
x_test, y_test = [], []
x_test_packed, y_test_packed = [], []

for family in families:

    f = open("../UPX/transformedOnTop20000/"+family+"-4000-frequency-Train.pickle","rb")
    partial_x = pickle.load(f)
    partial_x = np.asarray(partial_x)
    y_partial = [families.index(family)]*len(partial_x)

    if len(x_train) == 0:
        x_train = partial_x
        y_train = y_partial

    else:
        x_train = np.concatenate((x_train,partial_x))
        y_train = np.concatenate((y_train,y_partial))

    f = open("/media/ahmed/HDD/FamilyAnalysis/topK_dynamic/transformedOnTopKFeats/"+family+"-4000-frequency-Test.pickle","rb")
    partial_x = pickle.load(f)
    partial_x = np.asarray(partial_x)
    y_partial = [families.index(family)]*len(partial_x)
    if len(x_test) == 0:
        x_test = partial_x
        y_test = y_partial
    else:
        x_test = np.concatenate((x_test,partial_x))
        y_test = np.concatenate((y_test,y_partial))

    f = open("../UPX/transformedOnTop20000_unpacked/"+family+"-4000-frequency-Test.pickle","rb")
    partial_x = pickle.load(f)
    partial_x = np.asarray(partial_x)
    y_partial = [families.index(family)]*len(partial_x)
    if len(x_test_unpacked) == 0:
        x_test_unpacked = partial_x
        y_test_unpacked = y_partial
    else:
        x_test_unpacked = np.concatenate((x_test_unpacked,partial_x))
        y_test_unpacked = np.concatenate((y_test_unpacked,y_partial))


    f = open("../UPX/transformedOnTop20000/"+family+"-4000-frequency-Test.pickle","rb")
    partial_x = pickle.load(f)
    partial_x = np.asarray(partial_x)
    y_partial = [families.index(family)]*len(partial_x)
    if len(x_test_packed) == 0:
        x_test_packed = partial_x
        y_test_packed = y_partial
    else:
        x_test_packed = np.concatenate((x_test_packed,partial_x))
        y_test_packed = np.concatenate((y_test_packed,y_partial))

print(x_test_unpacked.shape)
print(y_test_unpacked.shape)
print(x_test_packed.shape)
print(y_test_packed.shape)

diff = []
for i in range(len(x_test_unpacked)):
    diff.append(abs(x_test_unpacked[i]-x_test_packed[i]))
    # diff.append(x_test_unpacked[i]-x_test_packed[i])
    # for j in range(len(x_test_unpacked[i])):
    #     diff[-1].append(abs(x_test_unpacked[i][j]-x_test_packed[i][j]))
diff = np.asarray(diff)
print(diff.shape)
AvgDiff = np.sum(diff,axis=0)
AvgDiff = AvgDiff/len(diff)
f = open("../UPXImpact.csv","w")
for i in range(len(AvgDiff)):
    f.write(str(i)+","+str(AvgDiff[i])+"\n")
f.close()
exit()
x_train_filtered = []
x_test_filtered = []
x_test_packed_filtered = []
x_test_unpacked_filtered = []

threshold = 0.2
AvgDiff = AvgDiff/max(AvgDiff)

for i in range(len(x_train)):
    x_train_filtered.append([])
    for j in range(len(x_train[i])):
        if AvgDiff[j] < threshold:
            x_train_filtered[-1].append(x_train[i][j])
x_train_filtered = np.asarray(x_train_filtered)
print(x_train_filtered.shape)

for i in range(len(x_test)):
    x_test_filtered.append([])
    for j in range(len(x_test[i])):
        if AvgDiff[j] < threshold:
            x_test_filtered[-1].append(x_test[i][j])
x_test_filtered = np.asarray(x_test_filtered)
print(x_test_filtered.shape)

for i in range(len(x_test_unpacked)):
    x_test_unpacked_filtered.append([])
    for j in range(len(x_test_unpacked[i])):
        if AvgDiff[j] < threshold:
            x_test_unpacked_filtered[-1].append(x_test_unpacked[i][j])
x_test_unpacked_filtered = np.asarray(x_test_unpacked_filtered)
print(x_test_unpacked_filtered.shape)

for i in range(len(x_test_packed)):
    x_test_packed_filtered.append([])
    for j in range(len(x_test_packed[i])):
        if AvgDiff[j] < threshold:
            x_test_packed_filtered[-1].append(x_test_packed[i][j])
x_test_packed_filtered = np.asarray(x_test_packed_filtered)
print(x_test_packed_filtered.shape)




print("==========================================================================")
print("RF")
print("==========================================================================")
clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train_filtered, y_train)
score = clf.score(x_test_filtered, y_test)
print("The Test score is:",score)
y_true = y_test
y_pred = clf.predict(x_test_filtered)
print("The Test F-1 is:",f1_score(y_true, y_pred, average='weighted'))
mat = confusion_matrix(y_true, y_pred)
matN = []
for i in range(len(mat)):
    matN.append(list(1.0*mat[i]/sum(mat[i])))
for i in range(len(matN)):
    for j in range(len(matN[i])):
        matN[i][j] = round(matN[i][j],3)
matN = np.asarray(matN)
print(matN)
df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
sns.set(font_scale=0.85)
ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
plt.savefig("../Heatmaps/FilteredTest.pdf")
plt.clf()
plt.close()


score = clf.score(x_test_unpacked_filtered, y_test_unpacked)
print("The Unpacked score is:",score)
y_true = y_test_unpacked
y_pred = clf.predict(x_test_unpacked_filtered)
print("The Unpacked F-1 is:",f1_score(y_true, y_pred, average='weighted'))
mat = confusion_matrix(y_true, y_pred)
matN = []
for i in range(len(mat)):
    matN.append(list(1.0*mat[i]/sum(mat[i])))
for i in range(len(matN)):
    for j in range(len(matN[i])):
        matN[i][j] = round(matN[i][j],3)
matN = np.asarray(matN)
print(matN)
df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
sns.set(font_scale=0.85)
ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
plt.savefig("../Heatmaps/FilteredUnpacked.pdf")
plt.clf()
plt.close()

score = clf.score(x_test_packed_filtered, y_test_packed)
print("The Packed score is:",score)
y_true = y_test_packed
y_pred = clf.predict(x_test_packed_filtered)
print("The Packed F-1 is:",f1_score(y_true, y_pred, average='weighted'))
mat = confusion_matrix(y_true, y_pred)
matN = []
for i in range(len(mat)):
    matN.append(list(1.0*mat[i]/sum(mat[i])))
for i in range(len(matN)):
    for j in range(len(matN[i])):
        matN[i][j] = round(matN[i][j],3)
matN = np.asarray(matN)
print(matN)
df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
sns.set(font_scale=0.85)
ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black')
plt.savefig("../Heatmaps/FilteredPacked.pdf")
plt.clf()
plt.close()
