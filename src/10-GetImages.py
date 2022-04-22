import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image

count = 1
path = "/media/ahmed/HDD/FamilyAnalysis/testing/testMalSamples/"
pathSave = "/media/ahmed/HDD/FamilyAnalysis/testing/mal2images/"
for root, dirs, files in os.walk(path):
    for filename in files:
        tmp = filename.replace("VirusShare_","")
        print("Malware",count)
        count += 1
        Filename = path+filename;
        f = open(Filename,'rb');
        ln = os.path.getsize(Filename);
        width = 256;
        rem = ln%width;
        a = array.array("B");
        a.fromfile(f,ln-rem);
        f.close();
        g = numpy.reshape(a,(int(len(a)/width),width));
        g = numpy.uint8(g);
        g = Image.fromarray(g.astype('uint8'))
        g = g.resize((256,256));
        g.save(pathSave+tmp+'.png')
