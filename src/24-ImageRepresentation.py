import requests
import json
import os
import subprocess
import cv2
import numpy
import pickle
width = 64
height = 64
path = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/testDataset/"
pathG = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/image/testDataset/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()


path = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/trainingMalData/"
pathG = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/image/trainingMalData/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()

path = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/upxPacked/"
pathG = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/image/upxPacked/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()



path = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/upxUnpacked/"
pathG = "/media/ahmed/HDD/FamilyAnalysis/priorWorks/binaries/image/upxUnpacked/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
