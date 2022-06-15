import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.python.keras.layers as tf_layers
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
from tensorflow.keras.layers.experimental.preprocessing import Normalization 
from pycomet.physics.thermochemistry.specie_thermo import partition_fun
from pycomet.physics.thermochemistry.specie_thermo import specie
from pycomet.utilities import utils


# =========================================================================== #
# TrainableAct Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='TrainableAct')
class TrainableAct(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        activation=None,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        stretch_axis={'x': False, 'y': True},
        pos_output=False,
        name='trainable_act',
        **kwargs
    ):
        super(TrainableAct, self).__init__(name=name, **kwargs)

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.stretch_axis = stretch_axis
        self.pos_output = pos_output

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                'A Dense layer can only be built with a floating-point '
                f'dtype. Received: dtype={dtype}'
            )
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                'The last dimension of the inputs to a Dense layer '
                'should be defined. Found None. '
                f'Full input shape received: {input_shape}'
            )

        if self.stretch_axis['y']:
            self.a = self.add_weight(
                'a',
                shape=[last_dim, last_dim],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True
            )

        if self.stretch_axis['x']:
            self.b = self.add_weight(
                'b',
                shape=[last_dim, last_dim],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True
            )

        self.built = True

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'stretch_axis': self.stretch_axis,
            'pos_output': self.pos_output
        }
        base_config = super(TrainableAct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        outputs = inputs

        # First matrix multiplication (x-axis stretching)
        if self.stretch_axis['x']:
            rank = outputs.shape.rank
            if rank == 2 or rank is None:
                outputs = tf.matmul(a=outputs, b=self.b)
            else:
                outputs = tf.tensordot(outputs, self.b, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = outputs.shape.as_list()
                output_shape = shape[:-1] + [self.b.shape[-1]]
                outputs.set_shape(output_shape)

        # Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Check if output is positive
        if self.pos_output and self.stretch_axis['y']:
            a = tf.math.exp(self.a)
        else:
            a = self.a

        # Second matrix multiplication (y-axis stretching)
        if self.stretch_axis['y']:
            rank = outputs.shape.rank
            if rank == 2 or rank is None:
                outputs = tf.matmul(a=outputs, b=a)
            else:
                outputs = tf.tensordot(outputs, a, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = outputs.shape.as_list()
                output_shape = shape[:-1] + [a.shape[-1]]
                outputs.set_shape(output_shape)

        return outputs


# =========================================================================== #
# Scaling Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='Scaling')
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

        self.weights_values = {
            'slope': self.slope,
            'offset': self.offset,
        }

    # Building
    ###########################################################################
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to "
                "`Scale` should be defined. Found `None`."
            )

        self._weights = {}
        for name, w in self.weights_values.items():
            self._weights[name] = self.add_weight(
                name=name,
                dtype=K.floatx(),
                initializer=initializers.Constant(value=w),
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
        return self.trans_fn(
            x * self._weights['slope'] + self._weights['offset']
        )


# =========================================================================== #
# Normalization Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package='PrODE', name='Normalization'
)
class Normalization(tf_layers.experimental.preprocessing.Normalization):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        file=None,
        columns=None,
        header=0,
        preproc_fn=None,
        mean=None,
        variance=None,
        axis=-1,
        **kwargs
    ):
        super(Normalization, self).__init__(
            axis=axis, mean=mean, variance=variance, **kwargs
        )
        if mean is None and variance is None:
            self.file = file
            self.columns = columns
            self.header = header
            self.preproc_fn = preproc_fn
            self.adapt(self.read_datafile(file))

    # Reading data from file --------------------------------------------------
    def read_datafile(self, file):
        file = tf.nest.flatten(file)
        data = []
        for f in file:
            utils.check_file_extension(f, '.csv')
            df = pd.read_csv(f, sep=',', header=self.header)
            df = df.to_numpy(dtype=K.floatx())
            if self.preproc_fn is not None:
                df = self.preproc_fn(df)
            data.append(self.read_dataframe(df, self.columns))
        data = np.vstack(data)
        return data

    def read_dataframe(self, df, col):
        if any(isinstance(c, (list,tuple)) for c in col):
            return [self.extract_data(df, c) for c in col]
        else:
            return self.extract_data(df, col)

    def extract_data(self, df, col):
        if len(col) == 2:
            indeces = np.arange(*col)
        elif len(col) > 2:
            indeces = col
        return np.take(df, indeces, axis=1)

    # def read_dataframe(self, df, col):
    #     if any(isinstance(i, (list,tuple)) for i in col):
    #         return [df[:,col[i][0]:col[i][1]] for i in range(len(col))]
    #     else:
    #         return df[:,col[0]:col[1]]


# =========================================================================== #
# LinearNorm Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='LinearNorm')
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
            **kwargs
        )


# =========================================================================== #
# Logarithmic Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='Log')
class Log(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        base=10.,
        slope=1.,
        offset=1.,
        name='log',
        **kwargs
    ):
        super(Log, self).__init__(name=name, **kwargs)

        self.base = base
        self.slope = slope
        self.offset = offset

        if self.base == 'e':
            self.scale = 1.
        else:
            self.scale = np.log(float(self.base))

        self.weights_values = {
            'slope': self.slope,
            'offset': self.offset,
            'scale': self.scale
        }

    # Building
    ###########################################################################
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to "
                "`Scale` should be defined. Found `None`."
            )

        self._weights = {}
        for name, w in self.weights_values.items():
            self._weights[name] = self.add_weight(
                name=name,
                dtype=K.floatx(),
                initializer=initializers.Constant(value=w),
                trainable=self.trainable
            )

        self.built = True

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'base': self.base,
            'slope': self.slope,
            'offset': self.offset
        }
        base_config = super(Log, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Calling
    ###########################################################################
    def call(self, x, training=None):
        y = tf.math.log(x + self._weights['offset']) / self._weights['scale']
        y = y + self._weights['slope']
        return y


# =========================================================================== #
# Bias Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='Bias')
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
            utils.raise_value_err(
                "The last dimension of the inputs to "
                "`Scale` should be defined. Found `None`."
            )

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
@tf.keras.utils.register_keras_serializable(package='PyCOMET', name='Equilibrium')
class Equilibrium(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        out_dim,
        species,
        reaction=None,
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

        for name, sp_ in self.species.items():
            if sp_.n_at > 1:
                self.molecule = name
            else:
                self.atom = name

        # Partition functions -------------------------------------------------
        if self.reaction is not None:
            self.intPF = {
                name: partition_fun.internal.Internal(specie) \
                    for name, specie in self.species.items()
            }
            if self.reaction == 'diss':
                self.transPF = {
                    name: partition_fun.translational.Translational(specie) \
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

    # Calling
    ###########################################################################
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x, T, rho = inputs
        else:
            utils.raise_value_err(
                "The inputs of the `{}` layer have to be a `list` "
                "or a `tuple`: [x, T]".format(self.name))

        if self.reaction is None:
            y = tf.split(x, num_or_size_splits=self.out_dim, axis=1)
            y = [ tf_layers.Softmax(axis=-1)(y_i) for y_i in y ]
        else:
            # Partition functions
            self.updatePF(T)
            if self.reaction == 'diss':
                Q = self.eq_diss(rho)
            elif self.reaction == 'excit':
                Q = [ pf.Q for pf in self.intPF.values() ]
                Q = tf_layers.Concatenate()(Q) if len(Q) > 1 else Q[-1]
            # Simil-Softmax layer
            y = tf_layers.Multiply()([tf.math.exp(x), Q])
            y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)
            y = [ y_i / tf.reduce_sum(y_i, axis=-1, keepdims=True) for y_i in y ]
            # y = tf_layers.Multiply()(
            #     [tf.math.exp(-x), Q / tf.reduce_sum(Q, axis=-1, keepdims=True)]
            # )
            # y = tf.split(y, num_or_size_splits=self.out_dim, axis=1)

        return y if len(y) > 1 else y[-1]

    def Keq(self):
        '''Diss.-Recomb. equilibrium constants matrix'''
        PFdot = {
            name: self.intPF[name].Q * self.transPF[name].Q \
                for name in self.species.keys()
        }
        return PFdot[self.atom]**2 / PFdot[self.molecule]

    def eq_diss(self, rho):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate equilibrium constant
        K = self.Keq()

        # Calculate mass fractions
        A = tf.math.pow(rho / self.species[self.atom].m, 2.)
        B = K * rho / self.species[self.molecule].m
        C = - B

        Y_O = (- B + tf.math.sqrt(B**2. - 4.*A*C)) / (2.*A)
        Y_O2 = 1. - Y_O

        return tf.concat([Y_O2, Y_O], 1)

    def updatePF(self, T):
        # Update partition functions
        for intPF_ in self.intPF.values():
            intPF_.update(T)
        if self.reaction == 'diss':
            for transPF_ in self.transPF.values():
                transPF_.update(T)
