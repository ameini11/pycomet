import os
import sys
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow_addons.image.interpolate_spline import _solve_interpolation
from tensorflow_addons.image.interpolate_spline import _apply_interpolation
from keras import layers as tf_layers
from keras.utils import tf_utils
from keras.utils import generic_utils
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import utils


# =========================================================================== #
# Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Layer(tf_layers.Layer):
    
    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        prefix=None,
        **kwargs
    ):
        self.prefix = prefix
        super(Layer, self).__init__(
            trainable,
            name,
            dtype,
            dynamic,
            **kwargs
        )
        
    def _init_set_name(self, name, zero_based=True):
        return _init_set_name(self, name, zero_based)

def _init_set_name(lay_cls, name, zero_based=True):
    if name is None:
        name = lay_cls.__class__.__name__
        if lay_cls.prefix is not None:
            name = lay_cls.prefix + '_' + name
        lay_cls._name = backend.unique_object_name(
            generic_utils.to_snake_case(name),
            zero_based=zero_based,
        )
    elif isinstance(name, str):
        backend.observe_object_name(name)
        lay_cls._name = name
    else:
        raise TypeError(
            f"Expected `name` argument to be a string, but got: {name}"
        )


# =========================================================================== #
# TrainableActivationFun Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class TrainableActivationFun(Layer):

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
# Interpolating Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Interpolating(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        file=None,
        columns=None,
        axis=1,
        train=None,
        order=3,
        regularization_weight=0.0,
        xformx=None, # xformx = {'dir': tf.math.log, 'inv': tf.math.exp}
        xformy=None,
        dim_out=2,
        **kwargs
    ):
        super(Interpolating, self).__init__(
            **kwargs
        )
        # Data from file
        self.file = file
        self.columns = columns
        self.axis = int(axis)
        # RBF 
        self.order = int(order)
        self.regularization_weight = float(regularization_weight)
        # Transformation
        condx = ((xformx is not None) and (not isinstance(xformx, dict)))
        condy = ((xformy is not None) and (not isinstance(xformy, dict)))
        if condx or condy:
            utils.raise_value_err(
                f"'xformx' and 'xformy' inputs passed to {self.name} "
                "must be a `dict` with keys ('dir', 'inv') for direct "
                "and inverse transformations."
            )
        self.xformx, self.xformy = xformx, xformy
        self.xform = {'x': self.xformx , 'y': self.xformy}
        # Dimensions
        self.dim_out = int(dim_out)
        # Training data
        self.get_train_data(train)

    # Interpolating functions
    ###########################################################################
    def get_train_data(self, train):
        if (train is not None) and (not isinstance(train, dict)):
            utils.raise_value_err(
                f"'train' input passed to {self.name} "
                "must be a `dict` with keys ('x', 'y')."
            )
        self.train = train
        if self.train is None:
            if os.path.isdir(self.file):
                self.file = f'{self.file}/{self.name}.txt'
            data = np.loadtxt(self.file)
            self.train = {}
            for k in ('x', 'y'):
                z = np.take(data, self.columns[k], axis=self.axis)
                if self.xform[k] is not None:
                    z = self.xform[k]['dir'](z)
                self.train[k] = np.array(z)

    def reshape_tensor(self, array, dim=3):
        tensor = tf.convert_to_tensor(array)
        shape = tuple(tensor.shape)
        if 1 <= dim <= 3:
            if int(dim - len(shape)) == 2:
                for i in (0,-1):
                    tensor = tf.expand_dims(tensor, i)
                return tensor
            elif int(dim - len(shape)) == 1:
                return tf.expand_dims(tensor, 0)
            elif int(dim - len(shape)) == 0:
                return tensor
            elif int(dim - len(shape)) == -1:
                return tf.squeeze(tensor, 0)
            elif int(dim - len(shape)) == -2:
                for i in (0,-1):
                    tensor = tf.squeeze(tensor, i)
                return tensor
        else:
            utils.raise_value_err(
                f'The input tensor in `{self.name}` is more than 3D.'
            )

    # Building
    ###########################################################################
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            utils.raise_value_err(
                "An Interpolating layer can only be built with a "
                f"floating-point dtype. Received: dtype={dtype}"
            )
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            utils.raise_value_err(
                "The last dimension of the inputs to an Interpolating layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        # Solve interpolation
        train_pts = self.reshape_tensor(self.train['x'], dim=3)
        rbf, lin = _solve_interpolation(
            train_points=train_pts,
            train_values=self.reshape_tensor(self.train['y'], dim=3),
            order=self.order,
            regularization_weight=self.regularization_weight
        )
        # Add weights
        for (k, v) in (("rbf", rbf), ("lin", lin), ("train_pts", train_pts)):
            w = self.add_weight(
                name=k,
                shape=v.shape,
                initializer=lambda shape, dtype: v,
                dtype=self.dtype,
                trainable=False
            )
            setattr(self, k, w)
        self.built = True

    # Calling
    ###########################################################################
    def call(self, inputs):
        y = inputs
        if self.xform['x'] is not None:
            y = self.xform['x']['dir'](y)
        y = self.apply_interp(y)
        if self.xform['y'] is not None:
            y = self.xform['y']['inv'](y)
        return y

    def apply_interp(self, query_points):
        query_points = self.reshape_tensor(query_points, dim=3)
        query_values = _apply_interpolation(
            query_points=query_points,
            train_points=self.train_pts,
            w=self.rbf, # Radial basis function term
            v=self.lin, # Linear term
            order=self.order
        )
        query_values = self.reshape_tensor(query_values, dim=self.dim_out)
        return query_values

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'file': self.file,
            'columns': self.columns,
            'axis': self.axis,
            'train': self.train,
            'order': self.order,
            'regularization_weight': self.regularization_weight,
            'xformx': self.xformx,
            'xformy': self.xformy,
            'dim_out': self.dim_out
        }
        base_config = super(Interpolating, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# =========================================================================== #
# Scaling Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Scaling(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        slope=1.,
        offset=0.,
        xform=None,
        **kwargs
    ):
        super(Scaling, self).__init__(
            **kwargs
        )
        self.slope = slope
        self.offset = offset
        self.xform = xform
        self.weights_values = {
            'slope': np.array(self.slope),
            'offset': np.array(self.offset)
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
                shape=w.shape,
                initializer=lambda shape, dtype: w,
                trainable=self.trainable
            )
        self.built = True

    # Calling
    ###########################################################################
    def call(self, inputs):
        outputs = inputs * self._weights['slope'] + self._weights['offset']
        if self.xform is not None:
            outputs = self.xform(outputs)
        return outputs

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'slope': self.slope,
            'offset': self.offset,
            'xform': self.xform
        }
        base_config = super(Scaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# ZeroToOneNormalization Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ZeroToOneNormalization(Scaling):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mn=0.0,
        mx=1.0,
        xform=None,
        file=None,
        axis=1,
        **kwargs
    ):
        if file is None:
            self.mn, self.mx = mn, mx
        else:
            self.mn, self.mx = np.split(np.loadtxt(file), 2, axis)
        # Convert to array
        for k in ("mn","mx"):
            x = getattr(self, k)
            x = np.array(x).reshape(-1)
            setattr(self, k, x)
        # Transform
        self.xform = xform
        if self.xform is not None:
            self.mn, self.mx = self.xform(self.mn), self.xform(self.mx)
        # Default values
        slope = 1.0 / self.mx
        offset = np.zeros_like(slope)
        # Avoid singularity
        idx = np.where(self.mx != self.mn)
        slope[idx] = 1.0 / (self.mx[idx]-self.mn[idx])
        offset[idx] = - self.mn[idx] * slope[idx]
        # Parent class
        super(ZeroToOneNormalization, self).__init__(
            slope=slope.reshape(1,-1),
            offset=offset.reshape(1,-1),
            **kwargs
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'mn': self.mn,
            'mx': self.mx,
            'xform': self.xform
        }
        base_config = super(ZeroToOneNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# # =========================================================================== #
# # Normalization Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(package=cfg._NAME)
# class StdNormalization(tf_layers.Normalization):

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         mean_var_file=None,
#         data_files=None,
#         columns=None,
#         header=0,
#         preproc_fn=None,
#         mean=None,
#         variance=None,
#         axis=-1,
#         save_dir='./saving/',
#         prefix=None,
#         name=None,
#         **kwargs
#     ):
#         self.mean_var_file = mean_var_file
#         self.data_files = data_files
#         self.columns = columns
#         self.header = header
#         self.preproc_fn = preproc_fn
#         self.save_dir = save_dir
#         self.prefix = prefix
#         if mean is None or variance is None:
#             _init_set_name(self, name)
#             self.saved_file = self.save_dir + '/' + self.name + '_mean_var.txt'
#             loaded = False
#             for f in (self.mean_var_file, self.saved_file):
#                 if (not loaded):
#                     if (f is not None) and (os.path.exists(f)):
#                         mean, variance = self.read_mean_var(f)
#                         loaded = True
#             if (not loaded) and self.data_files is not None:
#                 mean, variance = self.compute_mean_var()
#         super(StdNormalization, self).__init__(
#             axis=axis,
#             mean=mean,
#             variance=variance,
#             name=self.name,
#             **kwargs
#         )

#     # Handling mean and variance ----------------------------------------------
#     def compute_mean_var(self):
#         utils.print_fn(
#             f"Computing mean and variance of layer '{self.name}'", lev=1
#         )
#         data = self.read_datafile(self.data_files)
#         mean, variance = tf.nn.moments(data, axes=0)
#         os.makedirs(self.save_dir, exist_ok=True)
#         np.savetxt(self.saved_file, np.vstack((mean, variance)), fmt='%1.10e')
#         return mean, variance

#     def read_mean_var(self, file):
#         array = np.loadtxt(file)
#         mean, variance = array[0], array[1]
#         return mean, variance

#     # Reading data from file --------------------------------------------------
#     def read_datafile(self, files):
#         files = tf.nest.flatten(files)
#         data = []
#         for f in files:
#             utils.check_file_extension(f, '.csv')
#             df = pd.read_csv(f, sep=',', header=self.header)
#             df = df.to_numpy(dtype=backend.floatx())
#             if callable(self.preproc_fn):
#                 df = self.preproc_fn(df)
#             data.append(ops.extract_data_from_array(df, self.columns))
#         data = tf.convert_to_tensor(np.vstack(data))
#         return data

#     # Configuration update
#     ###########################################################################
#     def get_config(self):
#         config = {
#             'mean_var_file': self.mean_var_file,
#             'data_files': self.data_files,
#             'columns': self.columns,
#             'header': self.header,
#             'preproc_fn': self.preproc_fn,
#             'save_dir': self.save_dir,
#             'prefix': self.prefix
#         }
#         base_config = super(StdNormalization, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Normalization Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class StdNormalization(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mean_var_file=None,
        data_files=None,
        columns=None,
        header=0,
        preproc_fn=None,
        xform=None,
        mean=None,
        variance=None,
        axis=-1,
        save_dir='./saving/',
        **kwargs
    ):
        super(StdNormalization, self).__init__(
            **kwargs
        )
        self.mean_var_file = mean_var_file
        self.data_files = data_files
        self.columns = columns
        self.header = header
        self.preproc_fn = preproc_fn
        self.xform = xform
        self.save_dir = save_dir
        # Initialize normalization layer
        self.mean = mean
        self.variance = variance
        if self.mean is None or self.variance is None:
            self.saved_file = self.save_dir + '/' + self.name + '_mean_var.txt'
            loaded = False
            for f in (self.mean_var_file, self.saved_file):
                if (not loaded):
                    if (f is not None) and (os.path.exists(f)):
                        self.read_mean_var(f)
                        loaded = True
            if (not loaded) and self.data_files is not None:
                self.compute_mean_var()
        self.norm_lay = tf_layers.Normalization(
            axis=axis,
            mean=self.mean,
            variance=self.variance,
            trainable=self.trainable
        )

    # Handling mean and variance ----------------------------------------------
    def compute_mean_var(self):
        utils.print_fn(
            f"Computing mean and variance of layer '{self.name}'", lev=1
        )
        data = self.read_datafile(self.data_files)
        if self.xform is not None:
            data = self.xform(data)
        self.mean, self.variance = tf.nn.moments(data, axes=0)
        array = np.vstack((self.mean, self.variance))
        os.makedirs(self.save_dir, exist_ok=True)
        np.savetxt(self.saved_file, array, fmt='%1.10e')

    def read_mean_var(self, file):
        array = np.loadtxt(file)
        self.mean = array[0]
        self.variance = array[1]

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

    # Building
    ###########################################################################
    def build(self, input_shape):    
        return self.norm_lay.build(input_shape)
    
    # Calling
    ###########################################################################
    def call(self, inputs):
        if self.xform is not None:
            inputs = self.xform(inputs)
        return self.norm_lay.call(inputs)

    # Configuration update
    ###########################################################################
    def get_config(self):
        cfg = self.norm_lay.get_config()
        cfg['name'] = self.name
        cfg['xform'] = self.xform
        return cfg


# =========================================================================== #
# Logarithmic Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Log(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        base=10.0,
        slope=1.0,
        offset=0.0,
        **kwargs
    ):
        super(Log, self).__init__(
            **kwargs
        )
        self.base = base
        self.slope = slope
        self.offset = offset
        if self.base == 'e':
            self.scale = 1.0
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
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Bias(Layer):

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
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
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


# # =========================================================================== #
# # IOLambda Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(package=cfg._NAME)
# class IOLambda(Layer):

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         function,
#         signature=None,
#         output_shape=None,
#         mask=None,
#         arguments=None,
#         make_fn=True,
#         **kwargs
#     ):
#         super(IOLambda, self).__init__(
#             **kwargs
#         )
#         self.signature = signature
#         if self.signature is None:
#             self.signature = tuple([
#                 k for k in utils.get_fun_args(function) if k in ('x', 'y')
#             ])
#         if not any([k in ('x', 'y') for k in self.signature]):
#             utils.raise_value_err(
#                 f"The function signature passed into {self.name} accepts "
#                 "at least 'x', or 'y', or both as valid options, where 'x' "
#                 "<- model inputs and 'y' <- intermediate layer/model ouputs."
#             )

#         # Initialize Lambda layer
#         self.make_fn = make_fn
#         if self.make_fn:
#             function = self.make_function(function)
#             self.make_fn = False
#         self.lambda_lay = tf_layers.Lambda(
#             function=function,
#             output_shape=output_shape,
#             mask=mask,
#             arguments=arguments,
#             **kwargs
#         )

#     def make_function(self, fun):
#         def function(inputs, **kwargs):
#             inputs = {k: inputs[k] for k in self.signature}
#             return fun(**inputs, **kwargs)
#         return function

#     # Calling
#     ###########################################################################
#     def call(self, x=None, y=None, mask=None, training=None):
#         return self.lambda_lay.call(
#             inputs={'x': x, 'y': y}, mask=mask, training=training
#         )

#     # Configuration update
#     ###########################################################################
#     def get_config(self):
#         config = self.lambda_lay.get_config()
#         config.update({
#             'signature': self.signature,
#             'fun_made': self.fun_made
#         })
#         base_config = super(IOLambda, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         return cls.lambda_lay.from_config(
#             cls, config, custom_objects=custom_objects
#         )


# =========================================================================== #
# IOLambda Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
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


# =========================================================================== #
# Split Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Split(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        num_or_size_splits=1,
        axis=1,
        num=None,
        **kwargs
    ):
        super(Split, self).__init__(
            **kwargs
        )
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.num = num

    # Calling
    ###########################################################################
    def call(self, inputs, mask=None, training=None):
        return tf.split(
            inputs,
            num_or_size_splits=self.num_or_size_splits,
            axis=self.axis,
            num=self.num
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'num_or_size_splits': self.num_or_size_splits,
            'axis': self.axis,
            'num': self.num
        }
        base_config = super(Split, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# # =========================================================================== #
# # Concatenate Layer
# # =========================================================================== #
# @tf.keras.utils.register_keras_serializable(package=cfg._NAME)
# class Concatenate(tf_layers.Concatenate):

#     # Initialization
#     ###########################################################################
#     def __init__(
#         self,
#         axis=-1,
#         **kwargs
#     ):
#         super(Concatenate, self).__init__(
#             axis=-1,
#             **kwargs
#         )

#     # Calling
#     ###########################################################################
#     def call(self, inputs, training=None, mask=None):
#         y = inputs
#         if isinstance(y, (list,tuple)):
#             if len(y) > 1:
#                 return super(Concatenate, self).call(y)
#             else:
#                 return y[-1]
#         else:
#             return y


# =========================================================================== #
# Transform Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Transform(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        xform=None,
        **kwargs
    ):
        super(Transform, self).__init__(**kwargs)
        self.xform = xform

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        y = inputs
        if self.xform is not None:
            y = tf.nest.map_structure(self.xform, y)
        return y

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'xform': self.xform
        }
        base_config = super(Transform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# Limiter Layer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class Limiter(Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        limits=None,
        xform_in=None,
        xform_out=None,
        **kwargs
    ):
        super(Limiter, self).__init__(**kwargs)
        self.limits = limits
        # Functions
        self.xform_in = xform_in
        self.xform_out = xform_out
        # Transform layers
        self.xform_in_lay = Transform(self.xform_in)
        self.xform_out_lay = Transform(self.xform_out)

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
        self._weights = {
            f'lim_{i}': self.add_weight(
                name=f'lim_{i}',
                dtype=backend.floatx(),
                initializer=initializers.Constant(value=w),
                trainable=False
            ) for i, w in enumerate(self.limits)
        }
        self.built = True

    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        y = inputs
        y = self.xform_in_lay(y)
        y = tfp.math.clip_by_value_preserve_gradient(
            y, self._weights['lim_0'], self._weights['lim_1']
        )
        y = self.xform_out_lay(y)
        return y

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'limits': self.limits,
            'xform_in': self.xform_in,
            'xform_out': self.xform_out
        }
        base_config = super(Limiter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =========================================================================== #
# MultiLays Layer
# =========================================================================== #
class MultiLays(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        lays_propty,
        **kwargs
    ):
        super(MultiLays, self).__init__(
            **kwargs
        )
        # Order layers by index
        indices = list(lays_propty.keys())
        indices.sort()
        self.lays_propty = {i: lays_propty[i] for i in indices}
        # Initialize layers
        self.layers = {
            i: self.get_lay(propty) for i, propty in self.lays_propty.items()
        }

    # Initialize layers
    ###########################################################################
    def get_lay(self, propty={}):
        lay = propty['lay_id']
        if not callable(lay):
            lay = utils.get_class(
                modules=(sys.modules[__name__], tf_layers), name=lay
            )
        if 'kwargs' not in propty:
            propty['kwargs'] = {}
        return lay(**propty['kwargs'])

    # Calling
    ###########################################################################
    def __call__(self, inputs, training=None, *args, **kwargs):
        y = inputs
        for i, lay in self.layers.items():
            y = self.layers[i](y, training=training, *args, **kwargs)
        return y



# @tf.keras.utils.register_keras_serializable(package=cfg._NAME)
# class BaseRandomLayer(Layer):
#     """A layer handle the random number creation and savemodel behavior."""
    
#     def __init__(
#         self, seed=None, force_generator=False, rng_type=None, **kwargs
#     ):
#         """Initialize the BaseRandomLayer.

#         Note that the constructor is annotated with
#         @no_automatic_dependency_tracking. This is to skip the auto
#         tracking of self._random_generator instance, which is an AutoTrackable.
#         The backend.RandomGenerator could contain a tf.random.Generator instance
#         which will have tf.Variable as the internal state. We want to avoid
#         saving that state into model.weights and checkpoints for backward
#         compatibility reason. In the meantime, we still need to make them
#         visible to SavedModel when it is tracing the tf.function for the
#         `call()`.
#         See _list_extra_dependencies_for_serialization below for more details.

#         Args:
#           seed: optional integer, used to create RandomGenerator.
#           force_generator: boolean, default to False, whether to force the
#             RandomGenerator to use the code branch of tf.random.Generator.
#           rng_type: string, the rng type that will be passed to backend
#             RandomGenerator. Default to `None`, which will allow RandomGenerator
#             to choose types by itself. Valid values are "stateful", "stateless",
#             "legacy_stateful".
#           **kwargs: other keyword arguments that will be passed to the parent
#             *class
#         """
#         super().__init__(**kwargs)
#         self._random_generator = keras.backend.RandomGenerator(
#             seed, force_generator=force_generator, rng_type=rng_type
#         )

#     def build(self, input_shape):
#         super().build(input_shape)
#         self._random_generator._maybe_init()

#     def _trackable_children(self, save_type="checkpoint", **kwargs):
#         if save_type == "savedmodel":
#             cache = kwargs["cache"]
#             # TODO(b/213628533): This must be called before super() to ensure
#             # that any input shape changes are applied before getting the config
#             # of the model.
#             children = self._trackable_saved_model_saver.trackable_children(
#                 cache
#             )
#             # This method exposes the self._random_generator to SavedModel only
#             # (not layer.weights and checkpoint).
#             children["_random_generator"] = self._random_generator
#         else:
#             children = {}
#         children.update(super()._trackable_children(save_type, **kwargs))
#         return children

#     def _lookup_dependency(self, name):
#         # When loading from a Keras SavedModel load, make sure that the loader
#         # can find the random generator, otherwise the loader will assume that
#         # it does not exist, and will try to create a new generator.
#         if name == "_random_generator":
#             return self._random_generator
#         else:
#             return super()._lookup_dependency(name)


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class GaussianNoise(Layer):
    """Apply additive zero-centered Gaussian noise.

    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    Args:
      stddev: Float, standard deviation of the noise distribution.
      seed: Integer, optional random seed to enable deterministic behavior.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, stddev, seed=None, force_generator=False, rng_type=None, **kwargs):
        super().__init__(**kwargs)
        self._random_generator = keras.backend.RandomGenerator(
            seed, force_generator=force_generator, rng_type=rng_type
        )
        self.supports_masking = True
        self.stddev = stddev
        self.seed = seed

    def build(self, input_shape):
        super().build(input_shape)
        self._random_generator._maybe_init()

    def call(self, inputs, training=None):
        return backend.in_train_phase(self.noised, inputs, training=training)

    def noised(self, inputs):
        if isinstance(inputs, (list, tuple)):
            y = [self.randomgen(i) for i in inputs]
        elif isinstance(inputs, dict):
            y = {k: self.randomgen(i) for k, i in inputs.values()}
        else:
            y = self.randomgen(inputs)
        return y

    def randomgen(self, inputs):
        return inputs * (1 + self._random_generator.random_normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=self.stddev,
            dtype=inputs.dtype,
        ))

    def get_config(self):
        config = {"stddev": self.stddev, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
