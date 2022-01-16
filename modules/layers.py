# NN layers class
from abc import ABC, abstractmethod
import numpy as np

from activations import *


class ActivationLayer(object):
    '''
        Class for activatation function layer
        methods:
            forward()
            backward() 
    '''

    def __init__(self, name, activationFuncType=None):
        self.name = name
        if activationFuncType is None:
            self.activationFuncType = 'sigmoid'
        validFuncPool = ['sigmoid', 'relu']

        if self.activationFuncType.lower() not in validFuncPool:
            raise NameError(
                f"[Warning] {self.activationFuncType} not suportted")

    def activateFunction(self, input):
        '''
            get input and return the f(input)
        '''
        if self.activationFuncType == 'relu':
            return np.maximum(0, input)
        elif self.activationFuncType == 'sigmoid':
            return 1 / (1 + np.exp(-1 * input))

    def gradientOfFunction(self, input):
        '''
            return the grad of activation function
        '''
        grad = np.zeros(input.shape)
        if self.activationFuncType == 'sigmoid':  # x * (1 - x)
            x = self.activateFunction(input)
            return x * (1 - x)
        elif self.activationFuncType == 'relu':
            grad[input > 0] = 1
            grad[input < 0] = 0
            grad[input == 0] = 0  # in [0, 1] would be fine
            return grad

    def forward(self, input):
        return self.activateFunction(input)

    def backward(self, input):
        print('TODO')
        pass


class preActivationLayer(object):
    """
        Layer as NN node perception
        store weights, bias
        
        math: y = f(w^Tx + b)
    """

    def __init__(self, name, dim=(10, 10), initializationType='random'):
        '''
        param:
            dim: (input_size, out_size)
        '''
        self.name = name
        self.dim = dim

        # paramater of NN
        self.w = None
        self.b = None
        self.dw = None
        self.db = None

        self.initializeParamaters(initializationType)

    def initializeParamaters(self, initializationType):
        self.b = np.zeros(self.dim[1])

        if initializationType == 'random':
            self.w = np.random.random(self.dim)
            self.w /= np.sum(self.w)
            self.b = np.random.randn(self.dim[1])
        else:
            raise NameError(f'implement {initializationType}')

    def forward(self, input):
        return self.w.T @ input + self.b  # ???

    def backward(self, ):
        print("TODO")
        pass


# TODO: nn.linear layer


class LayerBase(ABC):

    def __init__(self, optimizer=None) -> None:
        self.X = []
        self.act_fn = None
        self.trainable = True
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


class Dense(LayerBase):
    """
        full connection layer
        Y = f(WX + b)
    """

    def __init__(self, out_size, act_fn=None, init='zero', optimizer=None):

        self.init = init
        self.n_in = None
        self.n_out = out_size
        self.act_fn = ActivationFunctions(act_fn)()

        self.params = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        # init_w = WeightInitializer(str(self.act_fn)) # TODO

        b = np.zeros((1, self.n_out))
        W = np.zeros((self.n_in, self.n_out))

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
