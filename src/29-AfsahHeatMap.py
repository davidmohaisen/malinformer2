import os, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Family = ['gafgyt', 'mirai', 'xorddos', 'tsunami', 'generica', 'ganiw', 'dofloo', 'setag', 'elknot', 'local']
f = open("/media/ahmed/HDD/FamilyAnalysis/topK_dynamic/famTopKFeatsInFam2-orderedByFamiliesInPaper.pickle","rb")
A = pickle.load(f)
for i in range(len(A)):
    for j in range(len(A[i])):
        A[i][j] = round(A[i][j]/1000,2)
        # A[i][j] = round(A[i][j]/1000.0,3)
        # A[i][j] = float(1.0*A[i][j]/1000.0)

matN = np.asarray(A)
print(matN)
df = pd.DataFrame(matN, columns=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"],index=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"])
sns.set(font_scale=0.85)
ax = sns.heatmap(df,annot=True,cmap="gray_r",cbar=False,linewidths=0.1, linecolor='black',fmt='g')
plt.savefig("../Heatmaps/famTopKFeatsInFam.pdf")
plt.clf()
plt.close()
