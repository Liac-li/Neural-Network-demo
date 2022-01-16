"""
  Collection of Activation functions
"""
from abc import ABC, abstractmethod
import numpy as np


class ActivationBase(ABC):

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        """
        apply activation function to input 
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)

        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        raise NotImplementedError


class Sigmoid(ActivationBase):

    def __init__(self, **kwargs):
        super().__init__()

    def __str__(self) -> str:
        return 'sigmoid'

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)

    def grad2(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x) * (1 - 2 * fn_x)


class ReLU(ActivationBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self) -> str:
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astrape(int)

    def grad2(self, x):
        return np.zeros_like(x)


class ActivationFunctions(object):

    def __init__(self, param):
        """
            param: str or instance of ActivationBase
        """
        self.param = param

    def __call__(self):
        if self.param is None:
            raise ValueError("Param can't be none")
        elif isinstance(self.param, ActivationBase):
            res = self.param
        elif isinstance(self.param, str):
            res = self.get_instance(self.param)
        else:
            raise ValueError(f"Unknown paramater: {self.param}")
        return res

    def get_instance(self, act_str) -> ActivationBase:
        act_str = act_str.lower()
        if act_str == 'sigmoid':
            act_fn = Sigmoid()
        elif act_str == 'relu':
            act_fn = ReLU()
        else:
            raise ValueError(f"Unknown activation: {act_str}")

        return act_fn
