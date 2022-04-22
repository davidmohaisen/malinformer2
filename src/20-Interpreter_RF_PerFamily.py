from sklearn.ensemble import RandomForestClassifier
import pickle
import pickle5
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
from sklearn.metrics import f1_score,precision_score,recall_score
from treeinterpreter import treeinterpreter as ti

warnings.filterwarnings("ignore")
families = ['mirai', 'xorddos', 'ganiw', 'tsunami', 'gafgyt', 'elknot', 'generica', 'setag', 'dofloo', 'local']


for family in families:
    # print(family)
    x_train, x_test, y_train, y_test = [], [], [], []

    for family_test in families:
        f = open("/media/ahmed/HDD/FamilyAnalysis/byFamilyTopFeatOccurrence_test-train/"+family+"-"+family_test+"-frequency-Train.pickle","rb")
        partial_x = pickle5.load(f)
        partial_x = np.asarray(partial_x[:,:20000])
        if family_test == family:
            y_partial = np.ones(len(partial_x))
        else:
            y_partial = np.zeros(len(partial_x))
        if len(x_train) == 0:
            x_train = partial_x
            y_train = y_partial

        else:
            x_train = np.concatenate((x_train,partial_x))
            y_train = np.concatenate((y_train,y_partial))

        f = open("/media/ahmed/HDD/FamilyAnalysis/byFamilyTopFeatOccurrence_test-train/"+family+"-"+family_test+"-frequency-Test.pickle","rb")
        partial_x = pickle5.load(f)
        partial_x = np.asarray(partial_x[:,:20000])

        if family_test == family:
            y_partial = np.ones(len(partial_x))
        else:
            y_partial = np.zeros(len(partial_x))
        if len(x_test) == 0:
            x_test = partial_x
            y_test = y_partial

        else:
            x_test = np.concatenate((x_test,partial_x))
            y_test = np.concatenate((y_test,y_partial))

    # for i in np.arange(500,20500,500):
    i = 20000
    x_train = np.asarray(x_train[:,:i])
    x_test = np.asarray(x_test[:,:i])
    # x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # x =  np.array(x)
    # clf = RandomForestClassifier().fit(x_train, y_train)
    # score = clf.score(x_test, y_test)
    # preds = clf.predict(x_test)
    # fscore = f1_score(y_test,preds)
    # precision = precision_score(y_test,preds)
    # recall = recall_score(y_test,preds)
    # print(family,i,"Accuracy:",score)
    # print(family,i,"F-1 Score:",fscore)
    # print(family,i,"Precision:",precision)
    # print(family,i,"Recall:",recall)
    # f = open("/media/ahmed/HDD/FamilyAnalysis/byFamilyTopFeatOccurrence_test-train/models/"+family+str(i),"wb")
    # pickle.dump(clf,f)
    f = open("/media/ahmed/HDD/FamilyAnalysis/byFamilyTopFeatOccurrence_test-train/models/"+family+str(i),"rb")
    clf = pickle.load(f)
    m = (clf.feature_importances_).argsort()[::-1]
    with open('../RFfeatureImportance/FeatureImportanceByRF-'+family+'.pickle', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(family)
    print(m)
    # exit()
    # prediction, bias, contributions = ti.predict(clf, x_test)
    # print(prediction.shape)
    # print(bias.shape)
    # print(contributions.shape)
    # exit()
