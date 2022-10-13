import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from keras import layers as tf_layers
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import utils


# =========================================================================== #
# TrainableActivationFun Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='TrainableActivationFun'
)
class TrainableActivationFun(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        activation=None,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        stretch_axis={'x': False, 'y': True},
        positive_output=False,
        **kwargs
    ):
        super(TrainableActivationFun, self).__init__(
            **kwargs
        )
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.stretch_axis = stretch_axis
        self.positive_output = positive_output

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            utils.raise_value_err(
                "Unable to build `TrainableActivationFun` layer with "
                f"non-floating point dtype {dtype}."
            )
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to `TrainableActivationFun` "
                "should be defined. Found `None`."
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

    # Calling
    ###########################################################################
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        outputs = inputs

        # First matrix multiplication (x-axis stretching) ---------------------
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
        if self.positive_output and self.stretch_axis['y']:
            a = tf.math.exp(self.a)
        else:
            a = self.a

        # Second matrix multiplication (y-axis stretching) --------------------
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

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'stretch_axis': self.stretch_axis,
            'positive_output': self.positive_output
        }
        base_config = super(TrainableActivationFun, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Scaling Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='Scaling'
)
class Scaling(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        slope=1.,
        offset=0.,
        trans_fn=None,
        **kwargs
    ):
        super(Scaling, self).__init__(
            **kwargs
        )
        self.slope = slope
        self.offset = offset
        self.trans_fn = trans_fn
        self.weights_values = {
            'slope': self.slope,
            'offset': self.offset,
        }

    # Building
    ###########################################################################
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to "
                f"`{self.name}` should be defined. Found `None`."
            )
        self._weights = {}
        for name, w in self.weights_values.items():
            self._weights[name] = self.add_weight(
                name=name,
                dtype=backend.floatx(),
                initializer=initializers.Constant(value=w),
                trainable=self.trainable
            )
        self.built = True

    # Calling
    ###########################################################################
    def call(self, inputs):
        outputs = inputs * self._weights['slope'] + self._weights['offset']
        if self.trans_fn is not None:
            outputs = self.trans_fn(outputs)
        return outputs

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


# =========================================================================== #
# LinearNormalization Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='LinearNormalization'
)
class LinearNormalization(Scaling):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mn=0.,
        mx=1.,
        **kwargs
    ):
        self.mn, self.mx = mn, mx
        if self.mx != self.mn:
            slope = 1. / (self.mx-self.mn)
            offset = - self.mn * slope
        else:
            slope = 1. / self.mx
            offset = 0.
        super(LinearNormalization, self).__init__(
            slope=slope,
            offset=offset,
            **kwargs
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'mn': self.mn,
            'mx': self.mx
        }
        base_config = super(LinearNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Normalization Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='Normalization'
)
class Normalization(tf_layers.Normalization):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mu_sigma_file=None,
        data_files=None,
        columns=None,
        header=0,
        preproc_fn=None,
        mean=None,
        variance=None,
        axis=-1,
        save_dir='./saving/',
        **kwargs
    ):
        super(Normalization, self).__init__(
            axis=axis, mean=mean, variance=variance, **kwargs
        )
        self.mu_sigma_file = mu_sigma_file
        self.data_files = data_files
        self.columns = columns
        self.header = header
        self.preproc_fn = preproc_fn
        self.save_dir = save_dir

    def build(self, input_shape):
        self.saved_file = self.save_dir + '/' + self.name + '_mu_sigma.txt'
        if self.input_mean is None and self.input_variance is None:
            if os.path.exists(self.mu_sigma_file):
                self.read_mu_sigma(file=self.mu_sigma_file)
            elif os.path.exists(self.saved_file):
                self.read_mu_sigma(file=self.saved_file)
            elif self.data_files is not None:
                self.compute_mu_sigma()
        super(Normalization, self).build(input_shape)

    # Handling mean and variance ----------------------------------------------
    def compute_mu_sigma(self):
        utils.print_fn(
            f"Computing mean and variance of layer '{self.name}'", lev=1
        )
        data = self.read_datafile(self.data_files)
        self.input_mean, self.input_variance = tf.nn.moments(data, axes=0)
        array = np.vstack((self.input_mean, self.input_variance))
        np.savetxt(self.saved_file, array, fmt='%1.10e')

    def read_mu_sigma(self, file):
        array = np.loadtxt(file)
        self.input_mean = array[0]
        self.input_variance = array[1]

    # Reading data from file --------------------------------------------------
    def read_datafile(self, file):
        file = tf.nest.flatten(file)
        data = []
        for f in file:
            utils.check_file_extension(f, '.csv')
            df = pd.read_csv(f, sep=',', header=self.header)
            df = df.to_numpy(dtype=backend.floatx())
            if callable(self.preproc_fn):
                df = self.preproc_fn(df)
            data.append(ops.extract_data_from_array(df, self.columns))
        data = tf.convert_to_tensor(np.vstack(data))
        return data

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'mu_sigma_file': self.mu_sigma_file,
            'data_files': self.data_files,
            'columns': self.columns,
            'header': self.header,
            'preproc_fn': self.preproc_fn,
            'save_dir': self.save_dir
        }
        base_config = super(Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Logarithmic Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='Log'
)
class Log(tf_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        base=10.,
        slope=0.,
        offset=1.,
        **kwargs
    ):
        super(Log, self).__init__(
            **kwargs
        )
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
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to "
                "`Log` should be defined. Found `None`."
            )
        self._weights = {}
        for name, w in self.weights_values.items():
            self._weights[name] = self.add_weight(
                name=name,
                dtype=backend.floatx(),
                initializer=initializers.Constant(value=w),
                trainable=self.trainable
            )
        self.built = True

    # Calling
    ###########################################################################
    def call(self, inputs):
        outputs = inputs + self._weights['offset']
        outputs = tf.math.log(self._weights['slope'] * outputs)
        return outputs / self._weights['scale']

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


# =========================================================================== #
# Bias Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='Bias'
)
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
        **kwargs
    ):
        super(Bias, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            utils.raise_value_err(
                "Unable to build `Bias` layer with "
                f"non-floating point dtype {dtype}."
            )
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to "
                "`Bias` should be defined. Found `None`."
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

    # Calling
    ###########################################################################
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        outputs = tf.nn.bias_add(inputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

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


# =========================================================================== #
# Permanent Dropout Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='Dropout'
)
class Dropout(tf_layers.Dropout):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        rate,
        noise_shape=None,
        seed=None,
        permanent=False,
        **kwargs
    ):
        super(Dropout, self).__init__(
            rate=rate,
            noise_shape=noise_shape,
            seed=seed,
            **kwargs
        )
        self.permanent = permanent

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        if self.permanent:
            training = True
        return super(Dropout, self).call(inputs, training)

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'permanent': self.permanent
        }
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# IOLambda Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='IOLambda'
)
class IOLambda(tf_layers.Lambda):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        function,
        signature=None,
        output_shape=None,
        mask=None,
        arguments=None,
        **kwargs
    ):
        self.signature = signature
        if self.signature is None:
            self.signature = tuple([
                k for k in utils.get_fun_args(function) if k in ('x', 'y')
            ])
        if not any([k in ('x', 'y') for k in self.signature]):
            utils.raise_value_err(
                f"The function signature passed into {self.name} accepts "
                "at least 'x', or 'y', or both as valid options, where 'x' "
                "<- model inputs and 'y' <- intermediate layer/model ouputs."
            )
        super(IOLambda, self).__init__(
            function=self.make_function(function),
            output_shape=output_shape,
            mask=mask,
            arguments=arguments,
            **kwargs
        )

    def make_function(self, fun):
        def function(inputs, **kwargs):
            inputs = {k: inputs[k] for k in self.signature}
            return fun(**inputs, **kwargs)
        return function

    # Calling
    ###########################################################################
    def call(self, x=None, y=None, mask=None, training=None):
        super(IOLambda, self).call(
            inputs={'x': x, 'y': y}, mask=mask, training=training
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'signature': self.signature
        }
        base_config = super(IOLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# # =========================================================================== #
# # IOLambda Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(
#     package=cfg._NAME, name='Split'
# )
# class Split(tf_layers.Lambda):

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         num_or_size_splits=1,
#         axis=1,
#         num=None,
#         output_shape=None,
#         mask=None,
#         arguments=None,
#         **kwargs
#     ):
#         self.num_or_size_splits = num_or_size_splits
#         self.axis = axis
#         self.num = num
#         super(Split, self).__init__(
#             function=self.make_function(),
#             output_shape=output_shape,
#             mask=mask,
#             arguments=arguments,
#             **kwargs
#         )

#     def make_function(self):
#         def function(x):
#             return tf.split(
#                 x,
#                 num_or_size_splits=self.num_or_size_splits,
#                 axis=self.axis,
#                 num=self.num
#             )
#         return function

#     # Configuration update
#     ###########################################################################
#     def get_config(self):
#         config = {
#             'num_or_size_splits': self.num_or_size_splits,
#             'axis': self.axis,
#             'num': self.num
#         }
#         base_config = super(Split, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
