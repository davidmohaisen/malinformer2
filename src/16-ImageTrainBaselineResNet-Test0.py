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
import pickle
from art.attacks import CarliniL2Method
from art.classifiers import KerasClassifier
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
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
def load_data(size):
    pathMalware = "../images/"
    x = []
    y = []
    f = open("../overall-fam_malw.pickle","rb")
    labels = pickle.load(f)
    counter = -1
    for i in labels:
        counter += 1
        for j in labels[i]:
            pic = Image.open(pathMalware+j+".png")
            pic = pic.resize(size);
            pix = np.array(pic)
            x.append(pix)

            y.append(counter)



    countSamples = []
    currentSamples = []
    for i in range(counter+1):
        countSamples.append(y.count(i))
        currentSamples.append(0)


    x = np.asarray(x)
    y = np.asarray(y)

    x = x.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    # x, y = shuffle(x, y, random_state=0)

    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    for i in range(len(x)):
        if currentSamples[y[i]] < 0.8*countSamples[y[i]]:
            currentSamples[y[i]]+= 1
            xtrain.append(x[i])
            ytrain.append(y[i])
        else:
            currentSamples[y[i]]+= 1
            xtest.append(x[i])
            ytest.append(y[i])
    xtrain = np.asarray(xtrain)
    ytrain = np.asarray(ytrain)
    xtest = np.asarray(xtest)
    ytest = np.asarray(ytest)
    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)
    return (xtrain, ytrain), (xtest, ytest)



def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

num_classes = 10

json_file = open('../Model/Image/Baseline.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../Model/Image/Baseline.h5")

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

(x_train, y_train), (x_test, y_test) = load_data(size=(64,64))
# (x_test, y_test) = load_test_data(size=(64,64))

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
