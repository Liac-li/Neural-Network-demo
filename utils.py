# Helper functions
import os
import gzip
import numpy as np
import copy


def load_minst_data(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    print(f'[Log] load local MNIST Data from {data_folder}')

    path = []
    for fname in files:
        path.append(os.path.join(data_folder, fname))

    with gzip.open(path[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(path[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    # flatten
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    y_train1 = []
    for i in y_train:
        tmp = np.zeros(10)
        tmp[i] = 1
        y_train1.append(tmp)
    y_test1 = [] 
    for i in y_test:
        tmp = np.zeros(10)
        tmp[i] = 1
        y_test1.append(tmp)


    return (x_train, np.array(y_train1)), (x_test, np.array(y_test1))


def mse_loss(y_true, y_pred):
    return np.sum(np.square(y_true-y_pred)) / y_true.shape[0]

def accuracy(y_true, y_pred):
    return (np.sum(y_true.argmax(axis=1) == y_pred.argmax(axis=1)) / y_true.shape[0])