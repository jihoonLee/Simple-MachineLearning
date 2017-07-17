import os
import numpy as np

learning_rate = 0.1
iteration = 10000

class SoftmaxRegression:
    def __init__(self, input_size, label_size):
        self.weight = np.ones((label_size, input_size), dtype = np.float)
        self.input_size = input_size
        self.label_size = label_size

    def get_output(self, x):
        wx = np.dot(self.weight, x.T)
        exp_wx = np.exp(wx)
        exp_sum = np.sum(exp_wx, axis=0)
        return np.divide(exp_wx, exp_sum)

    def get_prob(self, output, y):
        label_prob = output[y.flatten(), np.arange(output.shape[1])]
        return label_prob

    def get_cost(self, label_prob):
        cost = -np.log(label_prob)
        return np.mean(cost)

    def gradient_descent(self, x, y):
        m = x.shape[0]
        output = self.get_output(x)

        label_prob = self.get_prob(output, y)
        cost = self.get_cost(label_prob)
        grad = np.zeros((self.weight.shape))

        y_mat = np.zeros((output.shape))
        y_mat[y.flatten(), np.arange(output.shape[1])] = 1
        grad = learning_rate *  (1/m) * np.dot(y_mat-output, x)
        return grad, cost

    def weight_update(self, grad):
        self.weight += grad

    def test(self, x, y):
        count = 0.
        output = self.get_output(x)
        result = np.argmax(output, axis=0)
        return np.sum(result==y.flatten()) / y.shape[0]


def add_bias(arr):
    length = arr.shape[0]
    bias_arr = np.ones((length, 1))
    data = np.hstack((arr, bias_arr))
    return data

def load(path):
    train_data = np.genfromtxt(path+'/train_data.csv', delimiter=',', dtype=float)
    train_data = add_bias(train_data)
    train_label = np.genfromtxt(path+'/train_label.csv', delimiter=',', dtype=int)

    test_data = np.genfromtxt(path+'/test_data.csv', delimiter=',', dtype=float)
    test_data = add_bias(test_data)
    test_label = np.genfromtxt(path+'/test_label.csv', delimiter=',', dtype=int)

    return train_data, train_label, test_data, test_label


path = os.path.dirname(os.path.abspath(__file__)) +'/data'
train_data, train_label, test_data, test_label = load(path)

intput_size = train_data.shape[1]
output_size = 10
s_r = SoftmaxRegression(intput_size, output_size)

for i in range(1, iteration+1):
    diff, cost = s_r.gradient_descent(train_data, train_label)
    s_r.weight_update(diff)

    print ('%d iter  cost : %.4f ' % (i, cost))
    if i % 100 == 0:
        print ('>>>>>>> %d iter train acc : %.4f ' % (i, s_r.test(train_data, train_label)))
        print ('>>>>>>> %d iter test acc : %.4f ' % (i, s_r.test(test_data, test_label)))
