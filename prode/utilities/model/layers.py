import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers
import tensorflow.keras.layers as tf_layers

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from prode.physics.thermochemistry.specie import partitionFun
from prode.physics.thermochemistry.specie import specie
from prode.physics.general import const
from prode.utilities import utils


# =========================================================================== #
# MultiLayerBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='MultiLayerBlock')
class MultiLayerBlock(tf_layers.Layer):
    """Multi Layer Block."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        regularizer=None,
        weight_norm=False,
        batch_norm=None,
        dropout_rate=0,
        name='multi_layer_block',
        **kwargs
    ):
        super(MultiLayerBlock, self).__init__(name=name, **kwargs)
        
        ### Hyperparameters Initialization ------------------------------------
        self.units = units
        if batch_norm not in {None, "before", "after"}:
            utils.raise_value_err("Invalid `batch_norm` argument "
                "passed into `{}`. Available choices: {{None|'before'|"
                "'after'}}.".format(self.name))
        self.batch_norm = batch_norm
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer
        self.weight_norm = weight_norm
        self.dropout_rate = dropout_rate

        ### Layers Initialization ---------------------------------------------
        self.layers = []
        dense = tf_layers.Dense(
            units=self.units,
            activation=None if self.batch_norm == "before" else self.activation,
            use_bias=False if self.batch_norm == "before" else self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            name=self.name+'_dense'
        )
        if self.weight_norm:
            dense = tfa_layers.WeightNormalization(dense)
        self.layers.append(dense)

        if self.batch_norm is not None:
            self.layers.append(
                tf_layers.BatchNormalization(name=self.name+'_batchNorm')
            )

        if self.dropout_rate > 0:
            self.layers.append(
                tf_layers.Dropout(self.dropout_rate, name=self.name+'_dropout')
            )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'units': self.units,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'weight_norm': self.weight_norm,
            'dropout_rate': self.dropout_rate,
            'layers': self.layers
        }
        base_config = super(MultiLayerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        y = self.layers[0](x)
        if self.batch_norm == "before":
            y = self.activation(self.layers[1](y, training=training))
        elif self.batch_norm == "after":
            y = self.layers[1](y, training=training)
        if self.dropout_rate > 0:
            y = self.layers[-1](y, training=training)
        return y


# =========================================================================== #
# FFBlock Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='FFBlock')
class FFBlock(tf_layers.Layer):
    """Simple Feed-forward block."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        layer_size,
        activation,
        kernel_initializer,
        regularizer=None,
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
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        ### Layers Initialization ---------------------------------------------
        # Hidden Layers
        self.multi_lays = [
            MultiLayerBlock(
                units=self.layer_size[i],
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                name='hidden'+str(i-1)
            ) for i in range(1,len(self.layer_size)-1)
        ]

        # Output Layer
        self.multi_lays.append(
            MultiLayerBlock(
                units=self.layer_size[-1],
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                name='output'
            )
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'layer_size': self.layer_size,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'regularizer': self.regularizer,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'weight_norm': self.weight_norm,
            'multi_lays': self.multi_lays
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
        # Hidden/Output layers
        for i in range(len(self.layer_size)-1):
            y = self.multi_lays[i](y, training=training)
        return y


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
        regularizer=None,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='res_pi_block',
        **kwargs
    ):
        super(ResPIBlock, self).__init__(
            layer_size=layer_size,
            activation=activation,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer,
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
        self.U, self.V = [
            MultiLayerBlock(
                units=self.layer_size[1],
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                regularizer=self.regularizer,
                weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                name=name
            ) for name in ['U', 'V']
        ]

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
        u = self.U(y, training=training)
        v = self.V(y, training=training)
        # Hidden layers
        y = self.multi_lays[0](y, training=training)
        for i in range(1, len(self.layer_size) - 2):
            z = self.multi_lays[i](y, training=training)
            y = (1. - z) * u + z * v
        # Output layer
        y = self.multi_lays[-1](y, training=training)
        return y


# =========================================================================== #
# Scaling Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='Scaling')
class Scaling(tf_layers.Layer):

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

        self._slope = self.add_weight(
            name='slope',
            dtype=self.dtype,
            initializer=initializers.Constant(value=self.slope),
            trainable=self.trainable
        )

        self._offset = self.add_weight(
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
    def call(self, x, training=None):
        return self.trans_fn(x * self._slope + self._offset)


# =========================================================================== #
# LinearNorm Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='LinearNorm')
class LinearNorm(Scaling):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mn=0.,
        mx=1.,
        trans_fn=None,
        name='linear_norm',
        **kwargs
    ):
        if mx != mn:
            slope = 1. / (mx-mn)
            offset = - mn * slope
        else:
            slope = 1. / mx
            offset = 0.

        super(LinearNorm, self).__init__(
            slope=slope,
            offset=offset,
            name='scaling',
            **kwargs
        )


# =========================================================================== #
# Bias Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='Bias')
class Bias(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        activation=None,
        initializer='zeros',
        regularizer=None,
        constraint=None,
        activity_regularizer=None,
        name='bias',
        **kwargs
    ):
        super(Bias, self).__init__(
            activity_regularizer=activity_regularizer, name=name, **kwargs
        )

        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Bias` layer with non-floating"
                " point dtype %s." % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err("The last dimension of the inputs to "
                "`Bias` should be defined. Found `None`.")

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
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = math_ops.cast(x, dtype=self._compute_dtype_object)

        y = nn_ops.bias_add(x, self.bias)
        if self.activation is not None:
            y = self.activation(y)

        return y


# =========================================================================== #
# Equilibrium Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PrODE', name='Equilibrium')
class Equilibrium(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        out_dim,
        reaction,
        species,
        name='equilibrium',
        *args,
        **kwargs
    ):
        super(Equilibrium, self).__init__(
            name=name,
            *args,
            **kwargs
        )

        self.out_dim = out_dim
        self.reaction = reaction

        # Species -------------------------------------------------------------
        self.species = {
            sp: specie.Specie(sp, **info) if info != None else specie.Specie(sp) \
                for sp, info in species.items()
        }
    
        # Partition functions -------------------------------------------------
        self.intPF = {
            name: partitionFun.internalPF.Internal(specie) \
                for name, specie in self.species.items()
        }
        if self.reaction == 'diss':
            self.transPF = {
                name: partitionFun.translationalPF.Translational(specie) \
                    for name, specie in self.species.items()
            }

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'out_dim': self.out_dim,
            'reaction': self.reaction,
            'species': self.species
        }
        base_config = super(Equilibrium, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # # Calling
    # ###########################################################################
    # def call(self, inputs):
    #     if isinstance(inputs, (list, tuple)):
    #         x, T, rho = inputs
    #     else:
    #         utils.raise_value_err(
    #             "The inputs of the `{}` layer have to be a `list` "
    #             "or a `tuple`: [x, T]".format(self.name))

    #     if self.reaction == 'diss':
    #         y = tf.split(x, num_or_size_splits=self.out_dim, axis=1)
    #         y = [ tf_layers.Softmax(axis=-1)(y_i) for y_i in y ]
    #     else:
    #         # Partition function
    #         Q_int = [ pf.Q(T) for pf in self.intPF.values() ]
    #         Q = tf_layers.Concatenate()(Q_int) if len(Q_int) > 1 else Q_int[-1]

    #         # Simil Softmax layer
    #         y = tf_layers.Multiply()([tf.math.exp(x), Q])
    #         y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)
    #         y = [ y_i / tf.reduce_sum(y_i, axis=-1, keepdims=True) for y_i in y ]

    #     return y if len(y) > 1 else y[-1]

    # Calling
    ###########################################################################
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, T, rho = inputs
        else:
            utils.raise_value_err(
                "The inputs of the `{}` layer have to be a `list` "
                "or a `tuple`: [x, T]".format(self.name))

        # Partition functions -------------------------------------------------
        if self.reaction == 'diss':
            Q = self.eq_diss(rho, T)
        else:
            Q_int = [ pf.Q(T) for pf in self.intPF.values() ]
            Q = tf_layers.Concatenate()(Q_int) if len(Q_int) > 1 else Q_int[-1]

        # Simil Softmax layer
        y = tf_layers.Multiply()([tf.math.exp(x), Q])
        y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)
        y = [ y_i / tf.reduce_sum(y_i, axis=-1, keepdims=True) for y_i in y ]

        return y if len(y) > 1 else y[-1]

    def Keq_diss(self, T):
        '''Diss.-Recomb. equilibrium constants matrix'''
        num = tf.math.pow(
            tf.math.multiply(
                self.intPF['O'].Q(T), self.transPF['O'].Q(T)
            ), 2.
        )
        den = tf.math.multiply(
            self.intPF['O2'].Q(T), self.transPF['O2'].Q(T)
        )
        return num / den

    def eq_diss(self, rho, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate equilibrium constant
        K = self.Keq_diss(T)

        # Calculate mass fractions
        A = tf.math.pow(rho / self.species['O'].m, 2.)
        B = K * rho / self.species['O2'].m
        C = - B

        Y_O = (- B + tf.math.sqrt(B**2. - 4.*A*C)) / (2.*A)
        Y_O2 = 1. - Y_O

        return tf.concat([Y_O2, Y_O], 1)
