'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import matplotlib.pyplot as plt
import os
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
    return mat(dataMat), mat(labelMat).transpose()

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def batchGradAscent(dataMat, labelMat):
    m,n = shape(dataMat)
    alpha = 0.001
    maxCycles = 1
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
        print (weights)
    return weights

def classify(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def test():
    path = os.path.dirname(os.path.abspath(__file__)) + '/data0.csv'
    dataMat, labelMat = loadDataSet(path)
    weights0 = batchGradAscent(dataMat, labelMat)

if __name__ == '__main__':
    test()
