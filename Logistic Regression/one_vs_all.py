import os
import numpy as np

learning_rate = 0.1
iteration = 10000

class LogisticRegression:
    def __init__(self, input_size, label):
        self.weight = np.ones((1, input_size), dtype = np.float)
        self.label = label

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_output(self, x):
        h = np.dot(self.weight, x.T)
        return self.sigmoid(h)

    def get_error(self, x, y):
        output = self.get_output(x)
        return np.subtract(output, y==self.label)

    def gradient_descent(self, x, y):
        error = self.get_error(x, y)
        self.weight -= learning_rate * np.mean(np.multiply(error.T, x), axis=0)


def add_bias(arr):
    length = arr.shape[0]
    bias_arr = np.ones((length, 1))
    data = np.hstack((arr, bias_arr))
    return data

def load(path):
    train_data = np.genfromtxt(path+'/train_data.csv', delimiter=',', dtype=float)
    train_data = add_bias(train_data)
    train_label = np.genfromtxt(path+'/train_label.csv', delimiter=',', dtype=float)

    test_data = np.genfromtxt(path+'/test_data.csv', delimiter=',', dtype=float)
    test_data = add_bias(test_data)
    test_label = np.genfromtxt(path+'/test_label.csv', delimiter=',', dtype=float)

    return train_data, train_label, test_data, test_label

def test(classifier, test_data, test_label):
    count = 0.
    for x, y in zip(test_data, test_label):
        arg = np.zeros(10, dtype=np.float)
        for i in range(10):
            arg[i] = classifier[i].get_output(x)
        arg_index = np.unravel_index(arg.argmax(), arg.shape)[0]
        if y == arg_index:
            count+=1
    return count / test_label.shape[0]

path = os.path.dirname(os.path.abspath(__file__)) +'/data'
train_data, train_label, test_data, test_label = load(path)

train_size, intput_size = train_data.shape
logistic_arr = []
for i in range(10):
    logistic_arr.append(LogisticRegression(intput_size, i))

for i in range(iteration):
    for j in range(10):
        logistic_arr[j].gradient_descent(train_data, train_label)
    print ('%dth iteration acc >>  %f ' %(i, test(logistic_arr, test_data, test_label)))
