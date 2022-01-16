from ctypes import util
import numpy as np
from modules.activations import ActivationFunctions
from modules.layers import Dense, Softmax
from modules.losses import LossInitializer
import matplotlib.pyplot as plt

from utils import *
from modules.nn import MLP

mlp = MLP()
train_set, test_set = load_minst_data("../data-set")

x_train, y_train = train_set
x_test, y_test = test_set

mlp.sequence(
    Dense(32, act_fn='sigmoid', optimizer='sgd'),
    Dense(10, act_fn='sigmoid', optimizer='sgd'),
    Softmax(),
)

if __name__ == "__main__":
    args = {"epoch": 200, "loss": 'CrossEntropy'}

    loss_fn = LossInitializer(args['loss'])()
    last_act_fn = lambda x: x
    acc_list = []

    for i in range(args['epoch']):
        y_pred = mlp.forward(x_train)

        # y_pred = [np.argmax(item) for item in y_pred]
        # print(y_pred[100], y_train[100])

        acc = accuracy(y_train, y_pred)
        acc_list.append(acc)

        loss = loss_fn(y_train, y_pred)
        dy_main = loss_fn.grad(y_train, y_pred, y_pred, last_act_fn)

        mlp.backward(dy_main)

        print(f"[log] Epoch:{i} acc:{acc}, loss:{loss}")

    print("== Finish Training ==")
    y_test_pred = mlp.forward(x_test)
    acc = accuracy(y_test, y_test_pred)
    loss = loss_fn(y_test, y_test_pred)
    print(f"Test set acc: {acc}")
    print(f"Test {args['loss']} loss: {loss}")

    plt.plot(acc_list, 'bo-', label='acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('baseline + ReLU')
    plt.ylim((-0.1, 1.1))
    plt.show()
