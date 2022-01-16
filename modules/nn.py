from .layers import LayerBase


class MLP(object):

    def __init__(self, optimizer=None, init='zero') -> None:
        super().__init__()

        self.init = init
        self.optimizer = optimizer

        self._init_params()

    def _init_params(self):
        self._dv = {}  # dict of parmaters

    def sequence(self, *args):
        """
            Paramters:
            ---------------------------------
            kwargs: list of instance of layers 
        """

        self.layers = []

        for layer in args:
            if isinstance(layer, LayerBase):
                self.layers.append(layer)
            else:
                raise ValueError(f'layers must be LayerBase type')

    @property
    def parameters(self):
        raise NotImplementedError

    def forward(self, X_main):
        """
            Parameters:
            ---------------------------------
                X_main: input of all net work
        """
        layer_out = None
        cnt = 0
        for layer in self.layers:
            if layer_out is None:
                layer_out = layer.forward(X_main)
            else:
                layer_out = layer.forward(layer_out)

            self._dv[f"layer{cnt}"] = layer_out

        return layer_out  # final output

    def backward(self, dY_main=None):

        layer_back = None
        cnt = 0
        for layer in self.layers[::-1]:
            # print(str(layer))
            if layer_back is None:
                layer_back = layer.backward(dY_main)
            else:
                layer_back = layer.backward(layer_back)

            self._dv[f'dlayer{cnt}'] = layer_back
            layer.update()

        return layer_back
