# NN layers class
import numpy as np


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


class Dense(object):
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
