import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
f = open("../data/TSNE.pickle","rb")
components, Family = pickle.load(f)


UniqueFamily = list(set(Family))

number_of_colors = len(UniqueFamily)

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
# color = ["#000000","b", "g", "r", "c", "m", "y", ""]
x = []
y = []
z = []
for j in range(len(components)):
    x.append(components[j][0])
    y.append(components[j][1])
    z.append(components[j][2])

fig = plt.figure()
ax = plt.axes(projection='3d')

Marker = ["+", "^", "v", "o"]
Colors = ["b", "g", "r", "c", "k"]
for i in range(len(UniqueFamily)):
    print(UniqueFamily[i],(list(Family)).count(UniqueFamily[i]),Colors[int(i/4)],Marker[i%4])

ignoreList = ['gafgyt',"mirai",'tsunami']
ignoreList = []
for i in range(len(x)):
    if Family[i] in ignoreList:
        continue
    
    ax.scatter3D(x[i:i+1],y[i:i+1],z[i:i+1], c=Colors[int(UniqueFamily.index(Family[i])/4)],marker=Marker[UniqueFamily.index(Family[i])%4], s=30.0)
plt.show()
