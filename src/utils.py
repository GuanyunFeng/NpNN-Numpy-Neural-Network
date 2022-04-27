import os
import struct
import numpy as np

def load_mnist(path='../data/mnist', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels