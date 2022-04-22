import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE

f = open("../data/PCAComponents.pickle","rb")
Data, Family = pickle.load(f)

print(Data.shape)
# excludedFamilies = ["gafgyt","mirai","tsunami"]
excludedFamilies = []
x = []
fam = []
for i in range(len(Data)):
    if Family[i] in excludedFamilies:
        continue
    x.append(Data[i])
    fam.append(Family[i])
x = np.asarray(x)
fam = np.asarray(fam)
print(x.shape)

X_embedded = PCA(n_components=2).fit_transform(x)
print(X_embedded.shape)
f = open("../data/PCA2.pickle","wb")
pickle.dump([X_embedded,fam],f)
