import numpy as np
from scipy import ndimage
import cv2

def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy / fact * (X / fact) + Y / fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx / fact, sy / fact)
    return res


def resize(path, newpath):
    f = open(path)
    data, label = list(), list()
    size = 0
    for line in f.readlines():
        line = line.split(' ')
        label.append(line[-1])
        line = np.asarray(line[:len(line) - 1], dtype=np.float32)
        line = line.reshape(28, 28)
        line = block_mean(line, 2)
        line = line.astype('uint8')
        line = cv2.adaptiveThreshold(line, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        line /= 255
        line = line.reshape(14*14)
        data.append(line)
        size+=1

    b = np.zeros((size, 14*14+1))
    for i in xrange(len(data)):
        a = np.insert(data[i], 14*14, label[i])
        b[i] = a
    np.savetxt(newpath, b, delimiter=' ', fmt="%d")


if __name__ == '__main__':
    resize('/home/lee/Code/python/neuralnet/data/mnistTrain.txt', '/home/lee/Code/python/neuralnet/data/small_mnist_train.txt')
    #a = np.array([1, 2, 3])
    #print np.insert(a, 3, 4)