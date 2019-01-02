from layer import *
import os


class Network:
    def __init__(self, _iterations, _lr, data_set):
        self.layer_list = list()
        self.iterations = _iterations
        self.lr = _lr
        self.data_set = data_set

    def add_layer(self, layer):
        self.layer_list.append(layer)

    def set_target(self, train_target):
        target_layer = self.layer_list[-1]
        target_neurons = np.zeros(target_layer.size)
        target_neurons[int(train_target)] = 1
        target_layer.neurons = target_neurons

    def set_input(self, train_input):
        tmp_data = np.asarray(train_input, dtype=np.float32)
        self.layer_list[0].neurons = tmp_data

    def forward(self):
        for i in xrange(len(self.layer_list) - 2):
            prev_layer, next_layer = self.layer_list[i], self.layer_list[i + 1]
            next_layer.forward(prev_layer)

    def back(self):
        for i in reversed(xrange(1, len(self.layer_list)-1)):
            prev_layer, next_layer = self.layer_list[i], self.layer_list[i + 1]
            prev_layer.error_calculate(next_layer)

        for i in reversed(xrange(0, len(self.layer_list)-2)):
            prev_layer, next_layer = self.layer_list[i], self.layer_list[i + 1]
            next_layer.update_weight(prev_layer, self.lr)

    def train(self):
        print 'load train data'
        self.data_set.load_train()
        print 'train'
        for i in range(self.iterations):
            self.data_set.shuffle_train()
            if i != 0 and i % 10 == 0:
                self.test()

            for train_index in self.data_set.train_index:
                for k in train_index:
                    self.set_input(self.data_set.train_data[k])
                    self.set_target(self.data_set.train_target[k])
                    self.forward()
                    self.back()
            print str(i+1) + ' iter end'

    def predict(self):
        return self.layer_list[-2].neurons

    def test(self):
        count = 0.
        print 'load test data'
        self.data_set.load_test()
        self.data_set.shuffle_test()
        print 'test'
        for test_index in self.data_set.test_index:
            for k in test_index:
                self.set_input(self.data_set.test_data[k])
                self.forward()
                if int(self.data_set.test_target[k]) == np.argmax(self.predict()):
                    count += 1
        print 'Accuracy : %f' % (count / self.data_set.test_size)

    def network_builder(self):
        for i in xrange(len(self.layer_list) - 2):
            prev_layer, cur_layer = self.layer_list[i], self.layer_list[i + 1]
            cur_layer.init_weight(prev_layer)

    def save_network(self, path):
        if os.path.exists(path):
            os.remove(path)

        layers_info = list()
        for i in xrange(len(self.layer_list)):
            layer = self.layer_list[i]
            layers_info.append(layer.layer_info())

        with open(path, "wb") as f:
            pickle.dump(layers_info, f)

    def load_network(self, path):
        if os.path.exists(path) is False:
            print 'load file is not exist'
            return

        layers_load = list()

        with open(path, "rb") as f:
            layers_load = pickle.load(f)

        for i in xrange(len(self.layer_list)):
            layer = layers_load[i]
            layer_type = layer['layer_type']
            if layer_type != self.layer_list[i].layer_property:
                print 'Wrong Layer'
                return
            else:
                if layer_type == 1: # input
                    if layer['size'] != self.layer_list[i].size:
                        print 'Wrong Input Size'
                        return
                elif layer_type == 2: # hidden
                    if layer['size'] != self.layer_list[i].size:
                        print 'Wrong Hidden Size'
                        return
                    else:
                        self.layer_list[i].weight = layer['weight']
                elif layer_type == 3: # target
                    if layer['size'] != self.layer_list[i].size:
                        print 'Wrong Target Size'
                        return
                elif layer_type == 4: # output
                    if layer['size'] != self.layer_list[i].size:
                        print 'Wrong Output Size'
                        return
                    else:
                        self.layer_list[i].weight = layer['weight']
                elif layer_type == 5:
                    if layer['map_num'] != self.layer_list[i].map_num:
                        print 'Wrong Conv Map_num'
                        return
                    elif layer['height'] != self.layer_list[i].height:
                        print 'Wrong Conv Height'
                        return
                    elif layer['width'] != self.layer_list[i].width:
                        print 'Wrong Conv Width'
                        return
                    else:
                        self.layer_list[i].filter = layer['filter']
                elif layer_type == 6:
                    return
                elif layer_type == 7:
                    if layer['map_num'] != self.layer_list[i].map_num:
                        print 'Wrong Multi-Input Map_num'
                        return
                    elif layer['height'] != self.layer_list[i].height:
                        print 'Wrong Multi-Input Height'
                        return
                    elif layer['width'] != self.layer_list[i].width:
                        print 'Wrong Multi-Input Width'
                        return
