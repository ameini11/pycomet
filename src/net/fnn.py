import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers

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

        ### Layers Initialization ---------------------------------------------
        self.lays = list()
        # Hidden Layers
        for i in range(len(self.layer_size)-2):
            self.lays.append( self.hidden_block(i+1) )
        # Output Layer
        self.lays.append( self.dense(self.layer_size[-1], name='Output') )


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x

        # Input layer
        y = x
        if self._inp_trans is not None:
            y = self._inp_trans(y)

        # Hidden layers
        for i in range(len(self.layer_size) - 2):
            layer = self.lays[i]
            y = layer[0](y)
            if self.batch_norm == "before":
                y = self.activation( layer[0](y) )
            elif self.batch_norm == "after":
                y = layer[0](y)
            if self.dropout_rate > 0:
                y = layer[-1](y)

        # Output layer
        self.y = self.lays[-1](y)
        if self._out_trans is not None:
            self.y = self._out_trans(self.x, self.y)

        return self.y


    # Layers Definition
    ###########################################################################
    def hidden_block(self, number):

        act_fun = self.activation
        bias    = True
        if self.batch_norm == "before":
            act_fun = None
            bias    = False

        l = [ self.dense(self.layer_size[number], activation=act_fun, \
                         use_bias=bias, name='Hid'+str(number)) ]

        if self.batch_norm is not None:
            bn = lay.BatchNormalization(name='BNorm_Hid'+str(number))
            l.append(bn)

        if self.dropout_rate > 0:
            do = lay.Dropout(self.dropout_rate, noise_shape=None, \
                             training=self.dropout, name='Drop_Hid'+str(number))
            l.append(do)

        return l


    def dense(self, units, activation=None, use_bias=True, name=None):
        l = lay.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer="zeros",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name
        )
        if self.weight_norm:
            l = tfa.layers.WeightNormalization(l)

        return l
