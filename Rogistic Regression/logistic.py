import os
import numpy as np
import math
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x):
    if x > 0.5:
        return 1.
    else :
        return 0.

def load(path):
    csv = np.genfromtxt(path, delimiter='\t', dtype=float)
    length = csv.shape[0]
    data = csv[:, :2]
    label = csv[:, 2:3]
    tmp_arr = np.ones((length, 1))
    data = np.hstack((data, tmp_arr))
    return data, label

path = os.path.dirname(os.path.abspath(__file__)) + '/data0.csv'
data, label= load(path)
learning_rate = 0.01
m, n = data.shape
weight = np.ones((n, 1))

for i in range(10):
    acc = 0.
    for x, y in zip(data, label):
        h = sigmoid(np.sum(x * weight.transpose()))
        error = (h - y)
        print (error)
        weight = weight - (1./m * learning_rate * x.transpose() * error)
        if predict(h) == y:
            acc += 1
        break
    #print ('iter - %3d, acc - %.2f' % (i, acc/m))

"""
acc = 0.
for x, y in zip(test_data, test_label):
    h = sigmoid(np.sum(x * weight.transpose()))
    if predict(h) == y:
        acc += 1

print (acc/test_label.shape[0])
"""
