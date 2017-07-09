import os
import numpy as np
import math
import random

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def predict(x):
    if x > 0.5:
        return 1
    else :
        return 0

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataLine = [1.0]
        for i in lineArr:
            dataLine.append(float(i))
        label = dataLine.pop() # pop the last column referring to  label
        dataMat.append(dataLine)
        labelMat.append(int(label))
    return np.mat(dataMat), np.mat(labelMat).transpose()

path = os.path.dirname(os.path.abspath(__file__)) + '/data0.csv'
dataMat, labelMat = loadDataSet(path)
m,n = np.shape(dataMat)
weightMat = np.ones((n, 1), dtype=np.float)
learning_rate = 0.01

for i in range (10):
    acc = 0.
    for x, y in zip(dataMat, labelMat):
        h = sigmoid(x*weightMat)
        error = (h - y)
        print (error)
        weightMat = weightMat - (1./m * learning_rate * x.transpose() * error)
        break
        if predict(h) == y:
            acc += 1

    print ('iter - %3d, acc - %.2f' % (i, acc/m))
