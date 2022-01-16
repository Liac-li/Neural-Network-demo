# Helper functions
import os
import gzip
import numpy as np


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

    return (x_train, y_train), (x_test, y_test)


def mse_loss(y_true, y_pred):
    return np.sum(np.square(y_true-y_pred)) / y_true.shape[0]

def accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred) / y_true.shape[0])