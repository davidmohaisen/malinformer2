import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = []
varience = "0.999"
for i in np.arange(0,1000,100):
    f = open("../pca/PCA2components-family_variance-"+str(i)+"-"+varience+".pickle","rb")
    Data,_ = pickle.load(f)
    data.append([])
    for j in range(len(Data)):
        data[-1].append(Data[j])

data = np.asarray(data)
print(data.shape)
reshaped_data = data.reshape((1000,data.shape[-1]))
print(reshaped_data.shape)

X_embedded = TSNE(n_components=2,perplexity=3).fit_transform(reshaped_data)
X_embedded = X_embedded.reshape((10,100,X_embedded.shape[-1]))
print(X_embedded.shape)


Marker = ["^", "v"]
Colors = ["b", "g", "r", "k","c"]
for i in range(10):
    print(i,Colors[int(i/2)],Marker[i%2])


for i in range(len(X_embedded)):
    x = []
    y = []
    for j in range(len(X_embedded[i])):
        x.append(X_embedded[i][j][0])
        y.append(X_embedded[i][j][1])

    plt.scatter(x,y, c=Colors[int(i/2)],marker=Marker[i%2], s=5.0)
plt.show()
