from activation import *
import activation as act
import pickle
from scipy import signal

Layer_Property = {'input': 1, 'hidden': 2, 'target': 3, 'output': 4, 'conv': 5, 'pool': 6, 'multi-input': 7}
weight_store = None
class Layer:
    def __init__(self, layer_property, size=None, map_num=None, height=None, width=None):
        self._layer_property = Layer_Property[layer_property]
        if self._layer_property < 5:
            self._neurons = np.zeros(size)
            self._error = np.zeros(size)
            self._size = size
        else:
            self._neurons = np.zeros((map_num, height, width))
            self._error = np.zeros((map_num, height, width))
            self._size = map_num * height * width

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, data):
        self._error = data

    @property
    def layer_property(self):
        return self._layer_property

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, data):
        self._neurons = data

    @property
    def size(self):
        return self._size


class HiddenLayer(Layer):
    def __init__(self, size):
        Layer.__init__(self, 'hidden', size)
        self.weight = None

    def init_weight(self, prev_layer):
        self.weight = np.random.rand(self.size, prev_layer.size) - 0.5

    def error_calculate(self, next_layer):
        self.error = (np.tile(next_layer.error, (self.size, 1)).T * next_layer.weight).sum(axis=0) * derivative_sigmoid(self.neurons)

    def update_weight(self, prev_layer, lr):
        if prev_layer.layer_property == Layer_Property['conv']:
            self.weight += lr * (np.tile(self.error, (prev_layer.size, 1)) * np.tile(prev_layer.neurons.flatten(), (self.size, 1)).T).T
        else:
            self.weight += lr * (np.tile(self.error, (prev_layer.size, 1)) * np.tile(prev_layer.neurons, (self.size, 1)).T).T

    def forward(self, prev_layer):
        if prev_layer.layer_property == Layer_Property['conv']:
            self.neurons = act.sigmoid(self.weight.dot(prev_layer.neurons.flatten()))
        else:
            self.neurons = act.sigmoid(self.weight.dot(prev_layer.neurons))

    def layer_info(self):
        return {"layer_type" : self.layer_property, "size": self.size, "weight": self.weight}


class OutputLayer(Layer):
    def __init__(self, size):
        Layer.__init__(self, 'output', size)
        self.weight = None

    def init_weight(self, prev_layer):
        self.weight = np.random.rand(self.size, prev_layer.size) - 0.5

    def softmax(self):
        return self.size / np.sum(self._neurons)

    def error_calculate(self, next_layer):
        self.error = (next_layer.neurons - self.neurons) * derivative_sigmoid(self.neurons)

    def update_weight(self, prev_layer, lr):
        self.weight += lr * (np.tile(self.error, (prev_layer.size, 1)) * np.tile(prev_layer.neurons, (self.size, 1)).T).T

    def forward(self, prev_layer):
        self.neurons = act.sigmoid(self.weight.dot(prev_layer.neurons))

    def layer_info(self):
        return {"layer_type": self.layer_property, "size": self.size, "weight": self.weight}


class InputLayer(Layer):
    def __init__(self, input_num):
        Layer.__init__(self, 'input', input_num)

    def layer_info(self):
        return {"layer_type": self.layer_property, "size": self.size}


class MDInput(Layer):
    def __init__(self, width, height, feature_map_num):
        Layer.__init__(self, 'multi-input', map_num=feature_map_num, height=height, width=width)
        self.input_width = width
        self.input_height = height
        self.feature_map_num = feature_map_num

    @property
    def map_num(self):
        return self.feature_map_num

    @property
    def width(self):
        return self.input_width

    @property
    def height(self):
        return self.input_height

    def layer_info(self):
        return {"layer_type" : self.layer_property, "map_num": self.map_num, "height" : self.height, "width" : self.width}


class TargetLayer(Layer):
    def __init__(self, input_num):
        Layer.__init__(self, 'target', input_num)

    def layer_info(self):
        return {"layer_type" : self.layer_property, "size": self.size}

class ConvLayer(Layer):
    def __init__(self, width, height, feature_map_num, feature_size):
        Layer.__init__(self, 'conv', map_num=feature_map_num, height=height, width=width)
        self.input_width = width
        self.input_height = height
        self.feature_map_num = feature_map_num
        self.feature_size = feature_size
        self.conv_filter = None

    @property
    def map_num(self):
        return self.feature_map_num

    @property
    def width(self):
        return self.input_width

    @property
    def height(self):
        return self.input_height

    @property
    def filter(self):
        return self.conv_filter

    @filter.setter
    def filter(self, data):
        self.conv_filter = data

    def init_weight(self, prev_layer):
        self.conv_filter = np.random.rand(prev_layer.map_num, self.feature_map_num, self.feature_size,  self.feature_size) - 0.5

    def forward(self, prev_layer):
        prev_matrix = prev_layer.neurons
        for cur_feature in range(self.map_num):
            for prev_feature in range(prev_layer.map_num):
                self.neurons[cur_feature] += signal.convolve2d(prev_matrix[prev_feature], np.rot90(self.conv_filter[prev_feature][cur_feature], 2), mode='valid')
                self.neurons[cur_feature] = act.sigmoid(self.neurons[cur_feature])

    def error_calculate(self, next_layer):
        if next_layer.layer_property == Layer_Property['hidden']:
            self.error = ((np.tile(next_layer.error, (self.size, 1)).T * next_layer.weight).sum(axis=0) * derivative_sigmoid(self.neurons.flatten())).reshape(self.map_num, self.height, self.width)
        elif next_layer.layer_property == Layer_Property['pool']:
            return
        elif next_layer.layer_property == Layer_Property['conv']:
            self.error = np.zeros((self.map_num, self.height, self.width))
            for next_e in range(next_layer.map_num):
                for m in range(self.map_num):
                    self.error[m] += signal.convolve2d(next_layer.error[next_e], next_layer.filter[m][next_e], 'full')


    def update_weight(self, prev_layer, lr):
        filter_h = self.conv_filter.shape[2]
        filter_w = self.conv_filter.shape[3]

        for prev_m in range(prev_layer.map_num):
            for m in range(self.map_num):
                new_filter = np.zeros((filter_h, filter_w))
                kron_matrix = self.kron_error(self.error[m], filter_h, filter_w)
                for f_h in range(filter_h):
                    for f_w in range(filter_w):
                        new_filter[f_h, f_w] = self.matrix_sum(prev_layer.neurons[prev_m, f_h:f_h+self.height, f_w:f_w+self.width], kron_matrix[f_h * self.height: f_h * self.height + self.height, f_w * self.width:f_w * self.width + self.width])
                self.conv_filter[prev_m][m] += new_filter * lr

    def kron_error(self, data, height, width):
        return np.kron(data, np.ones((height, width)))

    def matrix_sum(self, matrix1, matrix2):
        return (matrix1.flatten() * matrix2.flatten()).sum()

    def layer_info(self):
        return {"layer_type": self.layer_property, "map_num": self.map_num, "height": self.height, "width": self.width, "feature_size": self.feature_size, "filter": self.filter}
