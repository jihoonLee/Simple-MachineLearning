import numpy as np
from pathlib import Path
import os

class Naive_Bayes:
    def __init__(self, input_size, label_size):
        self.input_size = input_size
        self.label_size = label_size

    def add_label_index(self, Y):
        label_index = np.hstack( (np.arange(Y.shape[0]).reshape(Y.shape), Y) )
        return label_index

    def get_count(self, X, label_index):
        likelihood = np.ones((self.label_size, self.input_size), dtype=np.float)
        priori = np.ones((self.label_size), dtype=np.float)

        for i in range(self.label_size):
            indices = (np.where(label_index[:,1]==i)[0])
            priori[i] += len(indices)
            total = priori[i] + 1
            likelihood[i] += np.sum(X[indices], axis=0)
            likelihood[i] /= total
        priori /= label_index.shape[0]
        return likelihood, priori

    def train(self, X, Y):
        label_idnex = self.add_label_index(Y)
        likelihood, priori = self.get_count(X, label_idnex)
        self.likelihood = np.log(likelihood)
        self.zero_likelihood = np.log(1-likelihood)
        self.priori = np.log(priori)

    def test(self, X, Y):
        predict = np.zeros( (X.shape[0], self.label_size) )
        for i in range(self.label_size):
            score = np.multiply (X, self.likelihood[i])
            score += np.multiply (X==0, self.zero_likelihood[i])
            score = np.sum(score, axis=1) + self.priori[i]
            predict[:, i] = score
        predict = (np.argmax(predict, axis=1) == Y.flatten())
        acc =  len(np.where(predict == True)[0]) / Y.shape[0]
        return acc

def load(path):
    train_data = np.genfromtxt(path+'/train_data.csv', delimiter=',', dtype=float)
    train_label = np.genfromtxt(path+'/train_label.csv', delimiter=',', dtype=float)

    test_data = np.genfromtxt(path+'/test_data.csv', delimiter=',', dtype=float)
    test_label = np.genfromtxt(path+'/test_label.csv', delimiter=',', dtype=float)
    return train_data, train_label.reshape(train_label.shape[0],1), test_data, test_label.reshape(test_label.shape[0],1)


path = Path(os.path.dirname(os.path.abspath(__file__))).parent
train_data, train_label, test_data, test_label = load(str(path) + '/data')

inpust_size = train_data.shape[1]
label_size = 10
nb = Naive_Bayes(inpust_size, label_size)
nb.train(train_data, train_label)
print ('train acc > %f' %(nb.test(train_data, train_label)))
print ('test acc > %f' %(nb.test(test_data, test_label)))
