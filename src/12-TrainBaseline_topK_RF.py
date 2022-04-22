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
warnings.filterwarnings("ignore")
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
    clf = RandomForestClassifier(criterion="entropy",random_state=0).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(num,score)
