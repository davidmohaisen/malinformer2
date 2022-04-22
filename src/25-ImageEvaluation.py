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
from sklearn.metrics import f1_score
FO = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']

path = "../priorWorks/binaries/image/"
insideFoldersBenign = ["testDataset/","upxPacked/","upxUnpacked/"]
labelsPath = ["testFamily_MalwareList.pickle","Packed-fam_malw_MalwareList.pickle","Packed-fam_malw_MalwareList.pickle"]

def load_data(pathToDealWith,labels):


    x_test = []
    y_test = []

    for sample in os.listdir(pathToDealWith):
        file_to_scan = pathToDealWith+sample
        f = open(file_to_scan,"rb")
        image = pickle.load(f)
        x_test.append(image)
        sample2 = sample.replace("VirusShare_","")
        for i in range(len(FO)):
            if sample in labels[FO[i]] or sample2 in labels[FO[i]]:
                y_test.append(i)
                flagAdded = True
                break
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print(x_test.shape)
    print(y_test.shape)
    return [x_test,y_test]


f = open(path+"trainFamily_MalwareList.pickle","rb")
labels = pickle.load(f)
(x_train, y_train) = load_data(pathToDealWith=path+"trainingMalData/",labels=labels)
x_train = x_train/255
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
y_train = keras.utils.to_categorical(y_train, 10)


model = Sequential()
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=20, shuffle=True)

# Score trained model.
scores = model.evaluate(x_train, y_train, verbose=1)
print('Test accuracy:', scores[1])


model_json = model.to_json()
with open("../Model/Image/Baseline.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../Model/Image/Baseline.h5")
print("Saved model to disk")
# exit()

results = ""
for i in range(len(insideFoldersBenign)):
    print(insideFoldersBenign[i])
    # results+="===============================\n"+insideFoldersBenign[i]+"\n"
    f = open(path+labelsPath[i],"rb")
    labels = pickle.load(f)
    (x_test, y_test) = load_data(pathToDealWith=path+insideFoldersBenign[i],labels=labels)
    x_test = x_test/255

    print(x_test.shape)
    print(y_test.shape)
    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
    y_true = y_test
    y_test = keras.utils.to_categorical(y_test, 10)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    y_pred = np.argmax(model.predict(x_test),axis = 1)
    print(y_pred)
    print(f1_score(y_true, y_pred, average='weighted'))

# print(results)
