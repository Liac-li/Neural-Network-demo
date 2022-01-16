from abc import ABC, abstractmethod

import numpy as np


class LossBase(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass


class SquardedError(LossBase):

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self) -> str:
        return "SquaredError"

    def loss(self, y_true, y_pred):
        return 0.5 * np.linalg.norm(y_pred - y_true)**2

    def grad(y_true, y_pred, z, act_fn):
        return (y_pred - y_true) * act_fn(z)


class CrossEntropy(LossBase):

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "CrossEntropy"

    def loss(self, y_true, y_pred):
        eps = np.finfo(float).eps

        cross_entropy = -np.sum(y_true * np.log(y_pred + eps))
        return cross_entropy

    def grad(self, y_true, y_pred, z, act_fn):
        grad = y_pred - y_true
        return grad
