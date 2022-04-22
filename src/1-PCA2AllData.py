import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc

# fileList = []
# for file in os.listdir("/media/ahmed/HDD/FamilyAnalysis/data/Ngrams/"):
#     if "frequency" in file:
#         fileList.append(file.rsplit("-")[0])
# print(len(fileList))
# fileList = list(set(fileList))
# fileList.sort()
# print(fileList)
# exit()

families = ['SINGLETON', 'ddostf', 'dnsamp', 'dofloo', 'elknot', 'ganiw', 'generica', 'goscanssh', 'httpsd', 'lady', 'local', 'mirai', 'rkit', 'setag', 'sshdoor', 'torii', 'tsunami', 'xmrig', 'znaich','gafgyt']
Data = []
Family = []
for fm in families:
    Freq = 0
    filePath = str("../../distinguishingPatternFinder/discriminativeFeats/interFamilyAnalysis/bySample/quad4NgramOccurrences/"+fm+"-frequency"+str(Freq)+".pickle")
    while os.path.exists(filePath):
        f = open(filePath,"rb")
        x = pickle.load(f)
        f.close()
        del f
        x[x>127] = 127

        x = np.array(x, dtype=np.byte)
        # for i in range(x.shape[0])
        print(filePath)
        print(x.shape, len(x))

        for y in x:
            Data.append(y)
            Family.append(fm)
        del x
        x = []
        Freq += 100
        filePath =  str("../../distinguishingPatternFinder/discriminativeFeats/interFamilyAnalysis/bySample/quad4NgramOccurrences/"+fm+"-frequency"+str(Freq)+".pickle")
        gc.collect(generation=0)
        if Freq == 500:
            break
print("Starting PCA")
print("length of Data ", len(Data))
print("length of Data ", len(Family))
pca = PCA(n_components = 2)
pca.fit(Data)
print("Fitting completed")

del Data
del Family
gc.collect(generation=0)


families = ['SINGLETON', 'ddostf', 'dnsamp', 'dofloo', 'elknot', 'ganiw', 'generica', 'goscanssh', 'httpsd', 'lady', 'local', 'mirai', 'rkit', 'setag', 'sshdoor', 'torii', 'tsunami', 'xmrig', 'znaich','gafgyt']

counter = 0
for fm in families:
    Freq = 0
    filePath = str("../../distinguishingPatternFinder/discriminativeFeats/interFamilyAnalysis/bySample/quad4NgramOccurrences/"+fm+"-frequency"+str(Freq)+".pickle")
    while os.path.exists(filePath):
        Data = []
        Family = []
        f = open(filePath,"rb")
        x = pickle.load(f)
        f.close()
        del f
        x[x>127] = 127

        x = np.array(x, dtype=np.byte)
        # for i in range(x.shape[0])
        print(filePath)
        print(x.shape, len(x))

        for y in x:
            Data.append(y)
            Family.append(fm)
        del x
        x = []
        Freq += 100
        filePath =  str("../../distinguishingPatternFinder/discriminativeFeats/interFamilyAnalysis/bySample/quad4NgramOccurrences/"+fm+"-frequency"+str(Freq)+".pickle")
        gc.collect(generation=0)
        reduced = pca.transform(Data)
        print("Transforming completed")
        print(reduced.shape)
        f = open("./PCA2components"+str(counter)+".pickle","wb")
        pickle.dump([reduced,Family],f)
        counter += 1
        del Data
        del Family
        del reduced
        gc.collect(generation=0)
