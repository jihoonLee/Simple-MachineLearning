from load.dataset import *
from net.network import *
from net.layer import *
if __name__ == '__main__':
    data_set = DataSet(1, 'data/Test.txt', 'data/Test.txt')
    nt = Network(500, 0.01, data_set)
    nt.add_layer(MDInput(14, 14, 1))
    nt.add_layer(ConvLayer(10, 10, 32, 5))
    nt.add_layer(ConvLayer(6, 6, 16, 5))
    nt.add_layer(HiddenLayer(50))
    nt.add_layer(OutputLayer(10))
    nt.add_layer(TargetLayer(10))
    nt.network_builder()
    # nt.load_network('data/save_network')
    nt.train()
    nt.test()
    nt.save_network('data/save_network1')
