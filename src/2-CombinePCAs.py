import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc

counter = 0
filePath = str("./components"+str(counter)+".pickle")
Data = []
Family = []

while os.path.exists(filePath):
    print(filePath)
    f = open("./components"+str(counter)+".pickle","rb")
    features, fam = pickle.load(f)

    for i in range(len(features)):
        Data.append(features[i])
        Family.append(fam[i])
    counter += 1
    filePath = str("./components"+str(counter)+".pickle")

Data = np.asarray(Data)
Family = np.asarray(Family)
print(Data.shape)
print(Family.shape)
f = open("./PCAComponents.pickle","wb")
pickle.dump([Data,Family],f)
