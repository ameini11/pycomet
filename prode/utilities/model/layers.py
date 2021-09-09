import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as lay

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from prode.utilities import utils


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
        name='ff_block',
        **kwargs
    ):
        super(FFBlock, self).__init__(name=name, **kwargs)
        
        ### Hyperparameters Initialization ------------------------------------
        self.layer_size = layer_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularization
        self.dropout_rate = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            utils.raise_value_err("Invalid `batch_norm` argument passed into `{}`. "
                "Available choices: {{None|'before'|'after'}}.".format(self.name))
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        self.lays = [self.DenseBlock(i) for i in range(1,len(self.layer_size)-1)]
        # Output Layer
        self.lays.append(self.Dense(self.layer_size[-1], name='output'))

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'layer_size':         self.layer_size,
            'activation':         self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer':        self.regularizer,
            'dropout_rate':       self.dropout_rate,
            'batch_norm':         self.batch_norm,
            'weight_norm':        self.weight_norm,
            'lays':               self.lays
        }
        base_config = super(FFBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):

        # Input layer
        y = x

        # Hidden layers
        for i in range(len(self.layer_size) - 2):
            y = self.call_block(y, self.lays[i], training=training)

        # Output layer
        y = self.lays[-1](y)

        return y

    def call_block(self, x, layer, training=None):
        y = layer[0](x)
        if self.batch_norm == "before":
            y = self.activation( layer[1](y) )
        elif self.batch_norm == "after":
            y = layer[1](y)
        if self.dropout_rate > 0:
            y = layer[-1](y, training=training)
        return y


    # Sub-layers Definition
    ###########################################################################
    def DenseBlock(
        self,
        index=1,
        name='hidden'
    ):
        lay_name = name + str(index-1)

        act_fn = None if self.batch_norm == "before" else self.activation
        bias = False if self.batch_norm == "before" else True

        lay_list = [self.Dense(
            self.layer_size[index],
            activation=act_fn,
            use_bias=bias,
            name=lay_name
        )]

        if self.batch_norm is not None:
            lay_list.append(
                lay.BatchNormalization(name=lay_name+'_batchNorm')
            )

        if self.dropout_rate > 0:
            lay_list.append(
                lay.Dropout(self.dropout_rate, name=lay_name+'_dropout')
            )

        return lay_list

    def Dense(
        self,
        units,
        activation=None,
        use_bias=True,
        name=None
    ):
        dense = lay.Dense(
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
            dense = tfa.layers.WeightNormalization(dense)

        return dense


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
        name='res_pi_block',
        **kwargs
    ):
        super(ResPIBlock, self).__init__(
            layer_size,
            activation,
            kernel_initializer,
            regularization=regularization,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            weight_norm=weight_norm,
            name=name,
            **kwargs
        )
        
        ### Hyperparameters Initialization ------------------------------------
        for i in range(1, len(self.layer_size)-2):
            if self.layer_size[i] != self.layer_size[i+1]:
                utils.raise_value_err(
                    "The number of neurons of each hidden layer must "
                    "be the same in `{}`.".format(self.name)
                )

        ### Layers Initialization ---------------------------------------------
        self.U = self.DenseBlock(name='U')
        self.V = self.DenseBlock(name='V')


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'U': self.U,
            'V': self.V
        }
        base_config = super(ResPIBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):

        # Input layer
        y = x

        # Transformer layers
        u = self.call_block(y, self.U, training=training)
        v = self.call_block(y, self.V, training=training)

        # Hidden layers
        y = self.call_block(y, self.lays[0], training=training)
        for i in range(1, len(self.layer_size) - 2):
            z = self.call_block(y, self.lays[i], training=training)
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
        slope=1.,
        offset=0.,
        trans_fn=None,
        name='scaling',
        **kwargs
    ):
        super(Scaling, self).__init__(name=name, **kwargs)

        self.slope = slope
        self.offset = offset

        if trans_fn is not None:
            self.trans_fn = trans_fn
        else:
            self.trans_fn = lambda x: x

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Scale` layer with non-floating"
                " point dtype %s." % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err("The last dimension of the inputs to "
                "`Scale` should be defined. Found `None`.")

        self.slope = self.add_weight(
            name='slope',
            dtype=self.dtype,
            initializer=initializers.Constant(value=self.slope),
            trainable=self.trainable
        )

        self.offset = self.add_weight(
            name='offset',
            dtype=self.dtype,
            initializer=initializers.Constant(value=self.offset),
            trainable=self.trainable
        )

        self.built = True

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'slope': self.slope,
            'offset': self.offset,
            'trans_fn': self.trans_fn
        }
        base_config = super(Scaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x):
        return self.trans_fn(x * self.slope + self.offset)


# =========================================================================== #
# Bias Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='BiasLay')
class BiasLay(lay.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        activation=None,
        initializer='zeros',
        regularizer=None,
        constraint=None,
        activity_regularizer=None,
        name='bias_lay',
        **kwargs
    ):
        super(BiasLay, self).__init__(
            activity_regularizer=activity_regularizer, name=name, **kwargs)

        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `BiasLay` layer with non-floating"
                " point dtype %s." % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err("The last dimension of the inputs to "
                "`BiasLay` should be defined. Found `None`.")

        self.bias = self.add_weight(
            'bias',
            shape=[last_dim,],
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            dtype=self.dtype,
            trainable=self.trainable
        )

        self.built = True

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
            'constraint': constraints.serialize(self.constraint)
        }
        base_config = super(BiasLay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        outputs = nn_ops.bias_add(inputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
