from requests import get
from scipy.ndimage.interpolation import zoom
import struct
import gzip
import numpy as np
import os

def download(url, file_name):
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

def unpack(fmt, file, read_size):
    value = struct.unpack(fmt, file.read(read_size))[0]
    return value

def unzip(src_image, src_label, dest_data, dest_label):
    f_label = gzip.open(src_label, 'rb')
    f_image = gzip.open(src_image, 'rb')
    f_label.read(4)
    f_image.read(4)
    NUM_IMAGE = unpack('>I', f_image, 4)
    ROWS = unpack('>I', f_image, 4)
    COLS = unpack('>I', f_image, 4)
    NUM_LABEL = unpack('>I', f_label, 4)
    if NUM_IMAGE != NUM_LABEL:
        raise Exception('did not match the number')
    else :
        N = NUM_IMAGE
    RESIZE = 14
    X = np.zeros((NUM_IMAGE, RESIZE*RESIZE),dtype=np.uint8)
    Y = np.zeros((NUM_LABEL, 1),dtype=np.uint8)

    for n in range(N):
        if n % 1000 == 0:
            print('Extract : %i' % (n))

        tmp_image = np.zeros((ROWS, COLS), dtype=np.uint8)
        for row in range(ROWS):
            for col in range(COLS):
                pixel = unpack('>B', f_image, 1)
                if pixel>128:
                    tmp_image[row, col] = 1
                else :
                    tmp_image[row, col] = 0

        tmp_image = zoom(tmp_image, RESIZE/ROWS)
        X[n] = tmp_image.reshape(RESIZE*RESIZE)
        Y[n] = unpack('>B', f_label, 1)

    np.savetxt(dest_data, X, delimiter=',', fmt='%d')
    np.savetxt(dest_label, Y, delimiter=',', fmt='%d')

path = os.path.dirname(os.path.abspath(__file__)) +'/data'
if not os.path.exists(path):
    os.makedirs(path)

download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', path + '/train-image.gz')
download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', path + '/train-label.gz')
download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', path + '/test-image.gz')
download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', path + '/test-label.gz')
unzip(path+'/train-image.gz', path+'/train-label.gz', path+'/train_data.csv', path+'/train_label.csv' )
unzip(path+'/test-image.gz', path+'/test-label.gz', path+'/test_data.csv', path+'/test_label.csv' )
