import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers
from . import layers
from . import steps

from .net import Net


class FNN(Net):
    """Feed-forward neural network."""

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


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'layer_size':         self.layer_size,
            'activation':         self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer':        self.regularizer,
            'dropout_rate':       self.dropout_rate,
            'batch_norm':         self.batch_norm,
            'weight_norm':        self.weight_norm,
            'block':              self.block,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x

        # Input Transform
        y = x
        if self.inp_trans is not None:
            y = lay.Lambda(self.inp_trans)(y)

        # Block Layer
        self.y = self.block(y)

        # Output Transform
        if self.out_trans is not None:
            self.y = lay.Lambda(self.out_trans)(self.y)

        return self.y


    # Training Step
    ##########################################################################
    def train_step(self, data):
        if self.res_loss_fn is not None:
            return steps.train_step(self, data)
        else:
            return super(FNN, self).train_step(data)


    # Testing Step
    ##########################################################################
    def test_step(self, data):
        if self.res_loss_fn is not None:
            return steps.test_step(self, data)
        else:
            return super(FNN, self).test_step(data)
