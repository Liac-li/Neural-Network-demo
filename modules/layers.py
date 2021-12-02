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
            raise NameError(f"[Warning] {self.activationFuncType} not suportted")
    
    def activateFunction(self, input):
        '''
            get input and return the f(input)
        '''
        if self.activationFuncType == 'relu':
           return np.maximum(0, input) 
        elif self.activationFuncType == 'sigmoid':
           return 1 / (1+np.exp(-1*input))
    
    def forward(self, input):
        return self.activateFunction(input)

    def backward(self, input):
        print("TODO")
        pass
