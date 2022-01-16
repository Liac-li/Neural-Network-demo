from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):

    def __init__(self, lr=0.01, **kwargs):
        super().__init__()
        self.lr = lr

        self.cache = {}
        self.cur_step = 0

    def __call__(self, params, grads):
        self.cur_step += 1

        return self.optimize(params, grads)

    @abstractmethod
    def optimize(self, params, grads):
        raise NotImplementedError


class SGD(OptimizerBase):

    def __init__(self, lr=0.01, **kwargs):
        super().__init__(lr=lr, **kwargs)

    def optimize(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

        return params


class Cycle(OptimizerBase):

    def __init__(self, lr=0.5, period=10, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.period = period

    def optimize(self, params, grads):
        lag = self.cur_step % self.period
        lr = self.lr / (1 + lag / self.period)

        for key in params.keys():
            params[key] -= lr * grads[key]

        return params


class OptimizerInitializer(object):

    def __init__(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.kwargs = kwargs

    def __call__(self):
        if self.optimizer_name is None:
            print('[Wargin]optimizer is None')
            return None

        if isinstance(self.optimizer_name, OptimizerBase):
            return self.optimizer_name

        # get from str
        if not isinstance(self.optimizer_name, str):
            raise ValueError(
                f'optimizer_name must be str or OptimizerBase type {self.optimizer_name}'
            )

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'sgd':
            return SGD(**self.kwargs)
        elif optimizer_name == 'cycle':
            return Cycle(**self.kwargs)
        else:
            raise NotImplementedError
