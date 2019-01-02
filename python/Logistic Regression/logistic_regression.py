import os
import numpy as np
from pathlib import Path
import math
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x):
    if x > 0.5:
        return 1.
    else :
        return 0.

def shuffle_data(pos, neg, size):
    pos_tmp = pos[np.random.choice(pos.shape[0], size=(size,))]
    neg_tmp = neg[np.random.choice(neg.shape[0], size=(size,))]
    shuffle = np.concatenate( (pos_tmp, neg_tmp), axis=0 )
    np.random.shuffle(shuffle)
    return shuffle[:, 2:], shuffle[:, 1]

def load(path):
    train_data = []
    train_label = []

    csv = np.genfromtxt(path, delimiter=',', dtype=float, skip_header=1)
    length = csv.shape[0]
    tmp_arr = np.ones((length, 1))
    csv = np.hstack((csv, tmp_arr))
    csv[:,2] = csv[:,2]/csv[:,2].max()
    csv[:,9] = csv[:,9]/csv[:,9].max()

    pos = csv[np.where(csv[:,1]==1.)]
    neg = csv[np.where(csv[:,1]==0.)]

    for i in range(20):
        data, label = shuffle_data(pos, neg, 25)
        train_data.append(data)
        train_label.append(label)
    test_data, test_label = shuffle_data(pos, neg, 50)
    return train_data, train_label, test_data, test_label

path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent) +'/data/KidCreative.csv'
train_data, train_label, test_data, test_label = load(path)
learning_rate = 0.001
m, n = train_data[0].shape
weight = np.ones(n)


for i in range(100000):
    if i%1000==0 and i!=0:
        print ('iteration : %d' %(i))

    for data, label in zip(train_data, train_label):
        acc = 0.
        weight_tmp = np.zeros(n)
        for x, y in zip(data, label):
            h = sigmoid(np.dot(x, weight))
            error = (h - y)
            weight_tmp += (1./m * learning_rate * x * error)
        weight = weight - weight_tmp

acc=0.
for x, y in zip(data, label):
    h = sigmoid(np.dot(x, weight))
    if predict(h) == y:
        acc += 1
print ('acc - %.2f' % (acc/m))
