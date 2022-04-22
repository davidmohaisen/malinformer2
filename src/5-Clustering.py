# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pickle


# #############################################################################
# Generate sample data
f = open("../data/PCAComponents.pickle","rb")
Data, Family = pickle.load(f)

# #############################################################################
# Compute DBSCAN
kmeans = AgglomerativeClustering(n_clusters=19).fit(Data)
print(kmeans.labels_[:50])
print(Family[:50])

print(kmeans.labels_[80:100])
print(Family[80:100])


print(kmeans.labels_[-50:])
print(Family[-50:])
