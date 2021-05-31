import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as lay

from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec


# =========================================================================== #
# FFBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='FFBlock')
class FFBlock(lay.Layer):
    """Simple Feed-forward block."""

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
        name='ff_block'):
        super(FFBlock, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size          = layer_size
        self.activation          = activation
        self.kernel_initializer  = kernel_initializer
        self.regularizer         = regularization
        self.dropout_rate        = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            utils.raise_value_err("Invalid `batch_norm` argument passed into `{}`. "
                "Available choices: {{None|'before'|'after'}}.".format(self.__name__))
        self.batch_norm          = batch_norm
        self.weight_norm         = weight_norm

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        self.lays = [ self.DenseBlock(number=i+1) for i in range(len(self.layer_size)-2) ]
        # Output Layer
        self.lays.append( self.Dense(self.layer_size[-1], name='output') )

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x):

        # Input layer
        y = x

        # Hidden layers
        for i in range(len(self.layer_size) - 2):
            y = self.call_block(y, self.lays[i])

        # Output layer
        y = self.lays[-1](y)

        return y

    def call_block(self, x, layer):
        y = layer[0](x)
        if self.batch_norm == "before":
            y = self.activation( layer[1](y) )
        elif self.batch_norm == "after":
            y = layer[1](y)
        if self.dropout_rate > 0:
            y = layer[-1](y)
        return y


    # Sub-layers Definition
    ###########################################################################
    def DenseBlock(self, number=None, name='hidden'):

        lay_name = name
        if number:
            lay_name = lay_name + str(number-1)

        act_fun = self.activation
        bias    = True
        if self.batch_norm == "before":
            act_fun = None
            bias    = False

        l = [ self.Dense(self.layer_size[number], activation=act_fun, \
                         use_bias=bias, name=lay_name) ]

        if self.batch_norm is not None:
            bn = lay.BatchNormalization(name=lay_name+'_batchNorm')
            l.append(bn)

        if self.dropout_rate > 0:
            do = lay.Dropout(self.dropout_rate, name=lay_name+'_dropout')
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
            name=name)
        
        if self.weight_norm:
            l = tfa.layers.WeightNormalization(l)

        return l


# =========================================================================== #
# ResPIBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='ResPIBlock')
class ResPIBlock(FFBlock):
    """An improved residual block architecture for PINNs.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

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
        name='res_pi_block'):
        super(ResPIBlock, self).__init__(
            layer_size,
            activation,
            kernel_initializer,
            regularization=regularization,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        for i in range(1, len(self.layer_size)-2):
            if self.layer_size[i] != self.layer_size[i+1]:
            utils.raise_value_err(
                "The number of neurons of each hidden layer must be the same in "
                "`{}`. Check `layer_size` variable.".format(self.__name__))

        ### Layers Initialization ---------------------------------------------
        self.U = self.DenseBlock(number=1, name='U')
        self.V = self.DenseBlock(number=1, name='V')


    # Configuration update
    ###########################################################################
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'U': self.U,
            'V': self.V,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def call(self, x):

        # Input layer
        y = x

        # Transformer layers
        u = self.U(x)
        v = self.V(x)

        # Hidden layers
        y = self.call_block(y, self.lays[0])
        for i in range(1,len(self.layer_size) - 2):
            z = self.call_block(y, self.lays[i])
            y = (1. - z) * u + z * v

        # Output layer
        y = self.lays[-1](y)

        return y


# =========================================================================== #
# Scaling Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='Scaling')
class Scaling(lay.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        init_scale=1.,
        init_offset=0.,
        trainable=True,
        trans_fn=None,
        name='scaling'):
        super(Scaling, self).__init__(name=name)

        self.init_scale  = init_scale
        self.init_offset = init_offset
        self.trainable   = trainable

        if trans_fn is not None:
            self.trans_fn = trans_fn
        else:
            self.trans_fn = lambda x: x

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Scale` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err("The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`.")
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

        self.scale = self.add_weight(
            name='scale',
            dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(value=self.init_scale),
            trainable=self.trainable)

        self.offset = self.add_weight(
            name='offset',
            dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(value=self.init_offset),
            trainable=self.trainable)

        self.built = True

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'init_scale':  self.init_scale,
            'init_offset': self.init_offset,
            'trainable':   self.trainable,
            'trans_fn':    self.trans_fn
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x):
        return self.trans_fn(x * self.scale + self.offset)
