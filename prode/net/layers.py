import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers


@tf.keras.utils.register_keras_serializable()
class FNNBlock(tf.keras.layers.Layer):

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
        name=None):
        super(FNNBlock, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size          = layer_size
        self.activation          = activation
        self.kernel_initializer  = kernel_initializer
        self.regularizer         = regularization
        self.dropout_rate        = dropout_rate
        self.batch_norm          = batch_norm
        self.weight_norm         = weight_norm

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        self.lays = [ self.DenseBlock(i+1) for i in range(len(self.layer_size)-2) ]
        # Output Layer
        self.lays.append( self.Dense(self.layer_size[-1], name=self.name+'_output') )


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x

        # Input layer
        y = self.x

        # Hidden layers
        for i in range(len(self.layer_size) - 2):
            layer = self.lays[i]
            y = layer[0](y)
            if self.batch_norm == "before":
                y = self.activation( layer[1](y) )
            elif self.batch_norm == "after":
                y = layer[1](y)
            if self.dropout_rate > 0:
                y = layer[-1](y)

        # Output layer
        self.y = self.lays[-1](y)

        return self.y
        

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
            'lays':               self.lays,
        })
        return config


    # Layers Definition
    ###########################################################################
    def DenseBlock(self, number):

        act_fun = self.activation
        bias    = True
        if self.batch_norm == "before":
            act_fun = None
            bias    = False

        l = [ self.Dense(self.layer_size[number], activation=act_fun, \
                         use_bias=bias, name=self.name+'_hid'+str(number)) ]

        if self.batch_norm is not None:
            bn = lay.BatchNormalization(name=self.name+'_normHid'+str(number))
            l.append(bn)

        if self.dropout_rate > 0:
            do = lay.Dropout(self.dropout_rate, name=self.name+'_dropHid'+str(number))
            l.append(do)

        return l


    def Dense(self, units, activation=None, use_bias=True, name=None):
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
