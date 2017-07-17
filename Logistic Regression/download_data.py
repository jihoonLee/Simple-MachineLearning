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

def extract(src_image, src_label, dest_data, dest_label, resize=False):
    print('%s %s' %(src_image, src_label))
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

    RESAMPLE = 14
    if resize:
        X = np.zeros((NUM_IMAGE, RESAMPLE*RESAMPLE),dtype=np.uint8)
    else:
        X = np.zeros((NUM_IMAGE, ROWS*COLS),dtype=np.uint8)
    Y = np.zeros((NUM_LABEL, 1),dtype=np.uint8)

    for n in range(N):
        if n % 1000 == 0:
            print('Extract : %i' % (n))

        image = np.zeros((ROWS*COLS), dtype=np.uint8)
        for row in range(ROWS):
            for col in range(COLS):
                pixel = unpack('>B', f_image, 1)
                if pixel>128:
                    image[row*ROWS+col] = 1
                else :
                    image[row*ROWS+col] = 0

        Y[n] = unpack('>B', f_label, 1)

        if resize:
            image = image.reshape(ROWS, COLS)
            image = zoom(image, zoom=0.5)
            X[n] = image.flatten()
        else:
            X[n] = image

    np.savetxt(dest_data, X, delimiter=',', fmt='%d')
    #np.savetxt(dest_label, Y, delimiter=',', fmt='%d')

path = os.path.dirname(os.path.abspath(__file__)) +'/data'
if not os.path.exists(path):
    os.makedirs(path)

#download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', path + '/train-image.gz')
#download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', path + '/train-label.gz')
#download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', path + '/test-image.gz')
#download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', path + '/test-label.gz')
extract(path+'/train-image.gz', path+'/train-label.gz', path+'/train_data_resize.csv', path+'/train_label.csv', resize=True )
extract(path+'/test-image.gz', path+'/test-label.gz', path+'/test_data_resize.csv', path+'/test_label.csv', resize=True )
