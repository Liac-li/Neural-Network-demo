# NN layers class
from abc import ABC, abstractmethod
import numpy as np

from .activations import *

class LayerBase(ABC):

    def __init__(self, optimizer=None, learning_rate=0.035) -> None:
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.learning_rate = learning_rate
        # self.optimizer = OptimizerInitialier(optimizer)()

        self.gradients = {}
        self.params = {}
        self.derived_variables = {}
        super().__init__()

    @abstractmethod
    def _init_params(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, X, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError
    
    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    
    def update(self):
        for param, grad in self.gradients.items():
            if param in self.params:
                self.params[param] -= self.learning_rate * grad
                # print(f"{param} updated, grad{grad}")
            self.gradients[param] = 0
        


class Dense(LayerBase):
    """
        full connection layer
        Y = f(WX + b)
    """

    def __init__(self, out_size, act_fn=None, init='zero', optimizer=None):
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = out_size
        self.act_fn = ActivationFunctions(act_fn)()

        self.params = {"W": None, "b": None}
        self.is_initialized = False
        
    def __str__(self):
        return "Dense: {} -> {}".format(self.n_in, self.n_out)

    def _init_params(self):
        # init_w = WeightInitializer(str(self.act_fn)) # TODO

        b = np.random.randn(*(1, self.n_out))
        W = np.random.randn(*(self.n_in, self.n_out))

        self.params = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y, Z = self._fwd(X)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        return Y

    def _fwd(self, X):
        W = self.params['W']
        b = self.params['b']

        Z = X @ W + b
        Y = self.act_fn(Z)
        return Y, Z

    def backward(self, dLdy, retain_grads=True):
        """
            Parmaters:
            dLdy:  <numpy.ndarray> as (n_ex, n_out)

                \delta in NN, dLdy_{i-1} = dLdy @ W @ act_fn.grad
        """

        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        W = self.params['W']
        b = self.params['b']

        Z = X @ W + b
        dZ = dLdy * self.act_fn.grad(Z)

        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)

        return dX, dW, dB

class Softmax(LayerBase):
    def __init__(self, dim=-1, optimizer=None):
        super().__init__(optimizer)
        
        self.dim = dim
        self.n_in = None
        self.is_initialized = False
        
    def _init_params(self):
        self.gradients = {}
        self.params = {}
        self.derived_variables = {}
        self.is_initialized = True
        
    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)
        
        if retain_derived:
            self.X.append(X)
        
        return Y
    
    def _fwd(self, X):
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)
    
    def backward(self, dLdy, retain_grads=True):
        
        if not isinstance(dLdy, list):
            dLdy = [dLdy]
            
        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)
        
        return dX[0] if len(X) == 1 else dX
    
    def _bwd(self, dLdy, X):
        dX = []
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T
                
                dxi.append(dyi @ dyidxi)
                
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)