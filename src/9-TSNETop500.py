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
    x =  np.array(x)
    print(x.shape)
    # exit()

    X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(x)
    X_embedded = X_embedded.reshape((len(families),100,X_embedded.shape[-1]))
    print(X_embedded.shape)
    # exit()


    Marker = ["^", "v"]
    Colors = ["b", "g", "r", "k","c"]
    for i in range(len(families)):
        print(families[i],Colors[int(i/2)],Marker[i%2])
    # exit()

    for i in range(len(X_embedded)):
        x = []
        y = []
        for j in range(len(X_embedded[i])):
            x.append(X_embedded[i][j][0])
            y.append(X_embedded[i][j][1])

        plt.scatter(x,y, c=Colors[int(i/2)],marker=Marker[i%2], s=4.0)
    # plt.show()
    plt.savefig("../"+str(num)+".pdf")
    plt.close()
