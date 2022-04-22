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
families = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']



x_test_unpacked, y_test_unpacked = [], []
x_train, y_train = [], []
x_test, y_test = [], []
x_test_packed, y_test_packed = [], []




print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("./RFBaselineModel.pickle","rb")
clf = pickle.load(f)

score = clf.score(x_test, y_test)
print("The Test score is:",score)
y_true = y_test
y_pred = clf.predict(x_test)
print("The Test F-1 is:",f1_score(y_true, y_pred, average='weighted'))
