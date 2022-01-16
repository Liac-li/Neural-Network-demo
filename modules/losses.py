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


class SquaredError(LossBase):

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self) -> str:
        return "SquaredError"

    def loss(self, y_true, y_pred):
        return 0.5 * np.linalg.norm(y_pred - y_true)**2

    def grad(self, y_true, y_pred, z, act_fn):
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


class LossInitializer(object):

    def __init__(self, param) -> None:
        self.param = param
        super().__init__()

    def __call__(self):
        if self.param is None:
            raise ValueError("Loss can't be none")
        elif isinstance(self.param, LossBase):
            res = self.param
        elif isinstance(self.param, str):
            res = self.get_instance(self.param)
        else:
            raise ValueError(f"Unknown loss function: {self.param}")

        return res

    def get_instance(self, loss_str):
        loss_str = loss_str.lower()
        if loss_str in 'MSE squarederror':
            loss_fn = SquaredError()
        elif loss_str in "crossentropy":
            loss_fn = CrossEntropy()
        else:
            raise ValueError(f"Unknown loss function:{loss_str}")

        return loss_fn
