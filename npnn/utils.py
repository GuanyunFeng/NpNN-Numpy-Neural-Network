import numpy as np
from struct import unpack
import gzip
import os

def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    return img

def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def load_mnist(path = "../data/mnist" , normalize=True, one_hot=True):
    
    '''读入MNIST数据集
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    Returns
    ----------
    (训练图像, 训练标签), (测试图像, 测试标签)
    '''
    image = {
        'train' : __read_image(os.path.join(path, "train-images-idx3-ubyte.gz")),
        'test'  : __read_image(os.path.join(path, "t10k-images-idx3-ubyte.gz"))
    }

    label = {
        'train' : __read_label(os.path.join(path, "train-labels-idx1-ubyte.gz")),
        'test'  : __read_label(os.path.join(path, "t10k-labels-idx1-ubyte.gz"))
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])

(x_train,y_train),(x_test,y_test)=load_mnist()