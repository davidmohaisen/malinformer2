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

FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

pathTrain = "../priorWorks/updatedData/trainingMalData/symbols/"
pathTest = "../priorWorks/updatedData/testDataset/symbols/"


x_test = []
y_test = []
f = open("../priorWorks/updatedData/testFamily_MalwareList.pickle","rb")
labels_test = pickle.load(f)


for fo_index in range(len(FO)):
    print(FO[fo_index],len(labels_test[FO[fo_index]]))
