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
nums = []
# families = set()
# for file in os.listdir('../noThreshold/'):
for file in os.listdir('../test_trainData/trainingData/'):

    num = int(file.rsplit("-")[0])
    # families.add(file.rsplit("-")[1])
    nums.append(num)
nums.sort()
print(nums)





def loadTestData(num):
    path = "/media/ahmed/HDD/FamilyAnalysis/test_trainData/testingData/"+str(num)+"-R2/"
    x_test = []
    y_test = []
    if not os.path.exists(path):
        return [],[]
    for file in os.listdir(path):
        if "frequency" not in file :
            continue

        f = open(path+file,"rb")
        Data = pickle.load(f)
        for j in range(len(Data)):
            x_test.append(Data[j])
            y_test.append(families.index(file.rsplit("-")[0]))
    return x_test,y_test

for num in nums:
    path = "../test_trainData/trainingData/"+str(num)+"-R2/"
    if not os.path.exists(path):
        continue

    x = []
    y = []
    for file in os.listdir(path):
        if "frequency" not in file:
            continue
        f = open(path+file,"rb")
        Data = pickle.load(f)
        # print(file,Data.shape)
        for j in range(len(Data)):
            x.append(Data[j])
            # print(file)
            # file = file.rsplit("/")[2]
            # print(file[2])
            y.append(families.index(file.rsplit("-")[0]))
    x = np.asarray(x)
    y = np.asarray(y)
    # print(num,x.shape)
    # print(num,y.shape)
    # exit()
    x_train, x_test, y_train, y_test = [], [], [], []
    total_y = collections.Counter(y)

    for i in range(len(x)):
        x_train.append(x[i])
        y_train.append(y[i])
    x_train, y_train = np.array(x_train), np.array(y_train)
    print(x_train.shape, y_train.shape)
    x =  np.array(x)
    clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)

    x_test, y_test = loadTestData(num)
    if len(x_test) == 0:
        continue
    score = clf.score(x_test, y_test)
    print(num,score)
    preds = clf.predict(x_test)
    con = confusion_matrix(y_test, preds)
    print(con)

    #######################################
    # x, y = loadTestData(num)
    # x_train, x_test, y_train, y_test = [], [], [], []
    # total_y = collections.Counter(y)
    # # print(total_y)
    # current_count = [0]*len(families)
    #
    # for i in range(len(x)):
    #     # split = 0.8*100
    #     if current_count[y[i]] >= 0.8*total_y[y[i]]:
    #         x_test.append(x[i])
    #         y_test.append(y[i])
    #         current_count[y[i]] += 1
    #     else:
    #         current_count[y[i]] += 1
    #         x_train.append(x[i])
    #         y_train.append(y[i])
    # x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    # clf = RandomForestClassifier(random_state=0).fit(x_trai
    # n, y_train)

    score = clf.score(x_test, y_test)
    print(num,score)
    preds = clf.predict(x_test)
    con = confusion_matrix(y_test, preds)
    print(con)
