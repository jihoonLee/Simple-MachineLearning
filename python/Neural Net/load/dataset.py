import numpy as np
import cv2
from random import shuffle
import random


class DataSet:
    def __init__(self, _batch, train_path, test_path):
        self._batch = _batch
        self.train_path = train_path
        self.test_path = test_path
        self.train_data, self.train_target, self.train_size = [], [], []
        self.test_data, self.test_target, self.test_size = [], [], []
        self.train_index = []
        self.test_index = []

    def load_train(self):
        self.train_data, self.train_target, self.train_size = self.read(self.train_path)

    def load_test(self):
        self.test_data, self.test_target, self.test_size = self.read(self.test_path)

    def read(self, path):
        f = open(path)
        size = 0
        data, label = list(), list()
        for line in f.readlines():
            size += 1
            line = line.split(' ')
            label.append(line[-1])
            line = np.asarray(line[:len(line) - 1], dtype=np.uint8)
            line = line.reshape(1, 14, 14)
            data.append(line)

        return data, label, size

    def shuffle_train(self):
        del self.train_index[:]
        self.shuffle(self.train_size, self.train_index)

    def shuffle_test(self):
        del self.test_index[:]
        self.shuffle(self.test_size, self.test_index)

    def shuffle(self, size, data_set):
        tmp_range = range(size)
        shuffle(tmp_range)
        tmp_index = []
        for i in tmp_range:
            tmp_index.append(i)
            if len(tmp_index) == self._batch:
                data_set.append(tmp_index)
                tmp_index = []
        while len(tmp_index) != self._batch:
            tmp_index.append(random.choice(tmp_range))
        data_set.append(tmp_index)

    @property
    def batch(self):
        return self._batchh

    @property
    def train_index(self):
        return self.train_index

    @property
    def test_index(self):
        return self.test_index

    @property
    def train_set(self):
        return self.train_data, self.train_target

    @property
    def test_index(self):
        return self.test_data, self.test_target

