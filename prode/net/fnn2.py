from . import activations
from . import initializers
from . import regularizers
from . import layers

from .net import Net


class FNN(Net):
    """Feed-forward neural networks."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='fnn'):
        super(FNN, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size          = layer_size
        self.activation          = activations.get(activation)
        self.kernel_initializer  = initializers.get(kernel_initializer)
        self.regularizer         = regularizers.get(regularization)
        self.dropout_rate        = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "batch_norm == {}. Available choices: {{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )
        self.batch_norm          = batch_norm
        self.weight_norm         = weight_norm

        self.name                = name

        ### Layers Initialization ---------------------------------------------
        self.block = layers.FNNBlock(
                        self.layer_size,
                        self.activation,
                        self.kernel_initializer,
                        regularization=self.regularizer,
                        dropout_rate=self.dropout_rate,
                        batch_norm=self.batch_norm,
                        weight_norm=self.weight_norm,
                        name=self.name)


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x

        # Input Transform
        y = x
        if self._inp_trans is not None:
            y = self._inp_trans(y)

        # Block Layer
        y = self.block(y)

        # Output Transform
        if self._out_trans is not None:
            self.y = self._out_trans(self.x, self.y)

        return self.y
