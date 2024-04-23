import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from keras.utils import tf_utils
from keras import layers as tf_layers
from keras.utils import generic_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.operations import linalg
from pycomet.optimization import losses
from pycomet.operations import interpolant
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Layer(tf_layers.Layer):

  # Initialization
  # ===================================
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
    if (name is None):
      name = utils.get_class_name(self)
      if (self.prefix is not None):
        name = self.prefix + '_' + name
      self._name = backend.unique_object_name(
        generic_utils.to_snake_case(name),
        zero_based=zero_based,
      )
    elif isinstance(name, str):
      backend.observe_object_name(name)
      self._name = name
    else:
      utils.raise_err(
        TypeError,
        f"Expected 'name' argument to be a string, but got: {name}"
      )


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class TrainableActivationFun(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    activation=None,
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    stretch_axis={'x': False, 'y': True},
    positive_output=False,
    **kwargs
  ):
    super(TrainableActivationFun, self).__init__(**kwargs)
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.stretch_axis = stretch_axis
    self.positive_output = positive_output

  # Building
  # ===================================
  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or bkd.get_floatx())
    if (not (dtype.is_floating or dtype.is_complex)):
      utils.raise_err(
        ValueError,
        "Unable to build 'TrainableActivationFun' layer with " \
        f"non-floating point dtype {dtype}."
      )
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to 'TrainableActivationFun' " \
        "should be defined. Found 'None'."
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
  # ===================================
  def call(self, inputs):
    if (inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype):
      inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
    outputs = inputs
    # First matrix multiplication (x-axis stretching)
    # -----------------------
    if self.stretch_axis['x']:
      rank = outputs.shape.rank
      if ((rank == 2) or (rank is None)):
        outputs = tf.matmul(a=outputs, b=self.b)
      else:
        outputs = tf.tensordot(outputs, self.b, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if (not tf.executing_eagerly()):
        shape = outputs.shape.as_list()
        output_shape = shape[:-1] + [self.b.shape[-1]]
        outputs.set_shape(output_shape)
    # Apply activation function
    if (self.activation is not None):
      outputs = self.activation(outputs)
    # Check if output is positive
    if (self.positive_output and self.stretch_axis['y']):
      a = tf.math.exp(self.a)
    else:
      a = self.a
    # Second matrix multiplication (y-axis stretching)
    # -----------------------
    if self.stretch_axis['y']:
      rank = outputs.shape.rank
      if ((rank == 2) or (rank is None)):
        outputs = tf.matmul(a=outputs, b=a)
      else:
        outputs = tf.tensordot(outputs, a, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if (not tf.executing_eagerly()):
        shape = outputs.shape.as_list()
        output_shape = shape[:-1] + [a.shape[-1]]
        outputs.set_shape(output_shape)
    return outputs

  # Configuration update
  # ===================================
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


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Interpolating(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    file=None,
    columns=None,
    axis=1,
    train=None,
    order=3,
    smoothing=0,
    xformx=None, # xformx = {'dir': tf.math.log, 'inv': tf.math.exp}
    xformy=None,
    rank=2,
    tol=1e-6,
    **kwargs
  ):
    super(Interpolating, self).__init__(**kwargs)
    # Data from file
    self.file = file
    self.columns = columns
    self.axis = int(axis)
    # Interpolant properties
    self.order = int(order)
    self.phi = interpolant.get_phi(self.order)
    self.smoothing = float(smoothing)
    # Transformation
    condx = ((xformx is not None) and (not isinstance(xformx, dict)))
    condy = ((xformy is not None) and (not isinstance(xformy, dict)))
    if (condx or condy):
      utils.raise_err(
        ValueError,
        f"'xformx' and 'xformy' inputs passed to {self.name} " \
        "must be a 'dict' with keys ('dir', 'inv') for direct " \
        "and inverse transformations."
      )
    self.xformx, self.xformy = xformx, xformy
    self.xform = {'x': self.xformx , 'y': self.xformy}
    # Dimensions
    self.rank = int(rank)
    # Training data
    self.get_train_data(train)
    # Interpolation accuracy
    self.tol = tol

  def get_train_data(self, train):
    if ((train is not None) and (not isinstance(train, dict))):
      utils.raise_err(
        ValueError,
        f"'train' input passed to {self.name} " \
        "must be a 'dict' with keys ('x', 'y')."
      )
    self.train = train
    if (self.train is None):
      if os.path.isdir(self.file):
        self.file = f'{self.file}/{self.name}.txt'
      data = np.loadtxt(self.file)
      self.train = {}
      for k in ('x', 'y'):
        z = np.take(data, self.columns[k], axis=self.axis)
        if (self.xform[k] is not None):
          z = self.xform[k]['dir'](z)
        self.train[k] = np.array(z)

  # Building
  # ===================================
  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or bkd.get_floatx())
    if (not (dtype.is_floating or dtype.is_complex)):
      utils.raise_err(
        ValueError,
        "An Interpolating layer can only be built with a " \
        f"floating-point dtype. Received: dtype={dtype}"
      )
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to an Interpolating layer " \
        "should be defined. Found None.\n" \
        f"Full input shape received: {input_shape}"
      )
    # Solve interpolation
    c = interpolant.to_rank(self.train['x'], rank=2)
    f = interpolant.to_rank(self.train['y'], rank=2)
    w, v = interpolant.solve(
      c, f, phi=self.phi, smoothing=self.smoothing
    )
    # Check interpolation
    y_pred = interpolant.apply(self.train['x'], c, self.phi, w, v)
    err = losses.mean_absolute_percentage_error(self.train['y'], y_pred)
    err = tf.math.reduce_mean(err)
    if (err > self.tol):
      utils.warning(
        "The mean relative interpolation error is higher " \
        "than the tolerance: %.5e % > %.5e %" % (err, self.tol)
      )
    # Add weights
    for (k, v) in (("w", w), ("v", v), ("c", c)):
      wi = self.add_weight(
        name=k,
        shape=v.shape,
        initializer=lambda shape, dtype: v,
        dtype=self.dtype,
        trainable=False
      )
      setattr(self, k, wi)
    self.built = True

  # Calling
  # ===================================
  def call(self, inputs):
    y = inputs
    if (self.xform['x'] is not None):
      y = self.xform['x']['dir'](y)
    y = interpolant.to_rank(y, rank=2)
    y = interpolant.apply(y, self.c, self.phi, self.w, self.v)
    y = interpolant.to_rank(y, rank=self.rank)
    if (self.xform['y'] is not None):
      y = self.xform['y']['inv'](y)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'file': self.file,
      'columns': self.columns,
      'axis': self.axis,
      'train': self.train,
      'order': self.order,
      'smoothing': self.smoothing,
      'xformx': self.xformx,
      'xformy': self.xformy,
      'rank': self.rank,
      'tol': self.tol
    }
    base_config = super(Interpolating, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Scaling(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    slope=1,
    offset=0,
    xform=None,
    **kwargs
  ):
    super(Scaling, self).__init__(**kwargs)
    self.slope = slope
    self.offset = offset
    self.xform = xform
    self.weights_values = {
      'slope': np.array(self.slope),
      'offset': np.array(self.offset)
    }

  # Building
  # ===================================
  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to " \
        f"'{self.name}' should be defined. Found 'None'."
      )
    self._weights = {}
    for (name, w) in self.weights_values.items():
      self._weights[name] = self.add_weight(
        name=name,
        dtype=bkd.get_floatx(),
        shape=w.shape,
        initializer=lambda shape, dtype: w,
        trainable=self.trainable
      )
    self.built = True

  # Calling
  # ===================================
  def call(self, inputs):
    outputs = inputs * self._weights['slope'] + self._weights['offset']
    if (self.xform is not None):
      outputs = self.xform(outputs)
    return outputs

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'slope': self.slope,
      'offset': self.offset,
      'xform': self.xform
    }
    base_config = super(Scaling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class ZeroToOneNormalization(Scaling):

  # Initialization
  # ===================================
  def __init__(
    self,
    mn=0.0,
    mx=1.0,
    xform=None,
    file=None,
    axis=1,
    **kwargs
  ):
    if (file is None):
      self.mn, self.mx = mn, mx
    else:
      self.mn, self.mx = np.split(np.loadtxt(file), 2, axis)
    # Convert to array
    for k in ("mn", "mx"):
      x = getattr(self, k)
      x = np.array(x).reshape(-1)
      setattr(self, k, x)
    # Transform
    self.xform = xform
    if (self.xform is not None):
      self.mn, self.mx = self.xform(self.mn), self.xform(self.mx)
    # Default values
    slope = 1/self.mx
    offset = np.zeros_like(slope)
    # Avoid singularity
    idx = np.where(self.mx != self.mn)
    slope[idx] = 1/(self.mx[idx]-self.mn[idx])
    offset[idx] = - self.mn[idx] * slope[idx]
    # Parent class
    super(ZeroToOneNormalization, self).__init__(
      slope=slope.reshape(1,-1),
      offset=offset.reshape(1,-1),
      **kwargs
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'mn': self.mn,
      'mx': self.mx,
      'xform': self.xform
    }
    base_config = super(ZeroToOneNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class StdNormalization(Layer):

  # Initialization
  # ===================================
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
    super(StdNormalization, self).__init__(**kwargs)
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
    if ((self.mean is None) or (self.variance is None)):
      self.saved_file = self.save_dir + '/' + self.name + '_mean_var.txt'
      loaded = False
      for f in (self.mean_var_file, self.saved_file):
        if (not loaded):
          if ((f is not None) and (os.path.exists(f))):
            self.read_mean_var(f)
            loaded = True
      if ((not loaded) and (self.data_files is not None)):
        self.compute_mean_var()
    self.norm_lay = tf_layers.Normalization(
      axis=axis,
      mean=self.mean,
      variance=self.variance,
      trainable=self.trainable
    )

  # Handling mean and variance
  # -----------------------------------
  def compute_mean_var(self):
    utils.print_fn(
      f"Computing mean and variance of layer '{self.name}'", lev=1
    )
    data = self.read_datafile(self.data_files)
    if (self.xform is not None):
      data = self.xform(data)
    self.mean, self.variance = tf.nn.moments(data, axes=0)
    array = np.vstack((self.mean, self.variance))
    os.makedirs(self.save_dir, exist_ok=True)
    np.savetxt(self.saved_file, array, fmt='%1.10e')

  def read_mean_var(self, file):
    array = np.loadtxt(file)
    self.mean = array[0]
    self.variance = array[1]

  # Reading data from file
  # -----------------------------------
  def read_datafile(self, file):
    file = tf.nest.flatten(file)
    data = []
    for f in file:
      utils.check_file_extension(f, '.csv')
      df = pd.read_csv(f, sep=',', header=self.header)
      df = df.to_numpy(dtype=bkd.get_floatx())
      if callable(self.preproc_fn):
        df = self.preproc_fn(df)
      data.append(ops.extract_data_from_tensor(df, self.columns))
    data = tf.convert_to_tensor(np.vstack(data))
    return data

  # Building
  # ===================================
  def build(self, input_shape):
    return self.norm_lay.build(input_shape)

  # Calling
  # ===================================
  def call(self, inputs):
    if (self.xform is not None):
      inputs = self.xform(inputs)
    return self.norm_lay.call(inputs)

  # Configuration update
  # ===================================
  def get_config(self):
    cfg = self.norm_lay.get_config()
    cfg['name'] = self.name
    cfg['xform'] = self.xform
    return cfg


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Log(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    base=10,
    slope=1,
    offset=0,
    scale=1,
    **kwargs
  ):
    super(Log, self).__init__(**kwargs)
    self.base = base
    self.slope = slope
    self.offset = offset
    self.scale = scale
    if (self.base != 'e'):
      self.scale = self.scale/np.log(float(self.base))
    self.weights_values = {
      'slope': ops.to_tensor(self.slope),
      'offset': ops.to_tensor(self.offset),
      'scale': ops.to_tensor(self.scale)
    }

  # Building
  # ===================================
  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to " \
        "'Log' should be defined. Found 'None'."
      )
    self._weights = {}
    for (name, w) in self.weights_values.items():
      self._weights[name] = self.add_weight(
        name=name,
        shape=w.shape,
        dtype=bkd.get_floatx(),
        initializer=lambda shape, dtype: w,
        trainable=self.trainable
      )
    self.built = True

  # Calling
  # ===================================
  def call(self, inputs):
    outputs = inputs + self._weights['offset']
    outputs = tf.math.log(self._weights['slope'] * outputs)
    return self._weights['scale'] * outputs

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'base': self.base,
      'slope': self.slope,
      'offset': self.offset
    }
    base_config = super(Log, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Bias(Layer):

  # Initialization
  # ===================================
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
  # ===================================
  def build(self, input_shape):
    dtype = tf.as_dtype(self.dtype or bkd.get_floatx())
    if (not (dtype.is_floating or dtype.is_complex)):
      utils.raise_err(
        ValueError,
        "Unable to build 'Bias' layer with " \
        f"non-floating point dtype {dtype}."
      )
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to " \
        "'Bias' should be defined. Found 'None'."
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
  # ===================================
  def call(self, inputs):
    if (inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype):
      inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
    outputs = tf.nn.bias_add(inputs, self.bias)
    if (self.activation is not None):
      outputs = self.activation(outputs)
    return outputs

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'activation': activations.serialize(self.activation),
      'initializer': initializers.serialize(self.initializer),
      'regularizer': regularizers.serialize(self.regularizer),
      'constraint': constraints.serialize(self.constraint)
    }
    base_config = super(Bias, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Dropout(tf_layers.Dropout):

  # Initialization
  # ===================================
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
  # ===================================
  def call(self, inputs, training=None):
    if self.permanent:
      training = True
    return super(Dropout, self).call(inputs, training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'permanent': self.permanent
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class IOLambda(tf_layers.Lambda):

  # Initialization
  # ===================================
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
    if (self.signature is None):
      self.signature = tuple([
        k for k in utils.get_fun_args(function) if k in ('x', 'y')
      ])
    if (not any([k in ('x', 'y') for k in self.signature])):
      utils.raise_err(
        ValueError,
        f"The function signature passed into {self.name} accepts " \
        "at least 'x', or 'y', or both as valid options, where 'x' " \
        "<- model inputs and 'y' <- intermediate layer/model ouputs."
      )
    super(IOLambda, self).__init__(
      function=self.make_fun(function),
      output_shape=output_shape,
      mask=mask,
      arguments=arguments,
      **kwargs
    )

  def make_fun(self, fun):
    def fun(inputs, **kwargs):
      inputs = {k: inputs[k] for k in self.signature}
      return fun(**inputs, **kwargs)
    return fun

  # Calling
  # ===================================
  def call(self, x=None, y=None, mask=None, training=None):
    super(IOLambda, self).call(
      inputs={'x': x, 'y': y}, mask=mask, training=training
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'signature': self.signature
    }
    base_config = super(IOLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Split(Layer):

  # Initialization
  # ===================================
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
  # ===================================
  def call(self, inputs, mask=None, training=None):
    return tf.split(
      inputs,
      num_or_size_splits=self.num_or_size_splits,
      axis=self.axis,
      num=self.num
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'num_or_size_splits': self.num_or_size_splits,
      'axis': self.axis,
      'num': self.num
    }
    base_config = super(Split, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Transform(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    xform=None,
    **kwargs
  ):
    super(Transform, self).__init__(**kwargs)
    self.xform = xform

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    y = inputs
    if (self.xform is not None):
      y = tf.nest.map_structure(self.xform, y)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'xform': self.xform
    }
    base_config = super(Transform, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Limiter(Layer):

  # Initialization
  # ===================================
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
  # ===================================
  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if (last_dim is None):
      utils.raise_err(
        ValueError,
        "The last dimension of the inputs to " \
        f"'{self.name}' should be defined. Found 'None'."
      )
    self._weights = {
      f'lim_{i}': self.add_weight(
        name=f'lim_{i}',
        dtype=bkd.get_floatx(),
        initializer=initializers.Constant(value=w),
        trainable=False
      ) for (i, w) in enumerate(self.limits)
    }
    self.built = True

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    y = inputs
    y = self.xform_in_lay(y)
    y = tfp.math.clip_by_value_preserve_gradient(
      y, self._weights['lim_0'], self._weights['lim_1']
    )
    y = self.xform_out_lay(y)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'limits': self.limits,
      'xform_in': self.xform_in,
      'xform_out': self.xform_out
    }
    base_config = super(Limiter, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GaussianNoise(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    stddev,
    seed=None,
    force_generator=False,
    rng_type=None,
    **kwargs
  ):
    super(GaussianNoise, self).__init__(**kwargs)
    self._random_generator = keras.backend.RandomGenerator(
      seed,
      force_generator=force_generator,
      rng_type=rng_type
    )
    self.supports_masking = True
    self.stddev = stddev
    self.seed = seed

  # Building
  # ===================================
  def build(self, input_shape):
    super(GaussianNoise, self).build(input_shape)
    self._random_generator._maybe_init()

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    return self.noised(inputs) if training else inputs

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

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "stddev": self.stddev,
      "seed": self.seed
    }
    base_config = super(GaussianNoise, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Gather(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    **kwargs
  ):
    super(Gather, self).__init__(**kwargs)
    # Indices
    self.indices = indices
    self.is_list = False
    if isinstance(self.indices[0], (list,tuple)):
      self.is_list = True
    # Extra arguments
    self.validate_indices = validate_indices
    self.axis = axis
    self.batch_dims = batch_dims
    self.kwargs = {
      "validate_indices": self.validate_indices,
      "axis": self.axis,
      "batch_dims": self.batch_dims
    }

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    if self.is_list:
      y = [tf.gather(inputs, indices=i, **self.kwargs) for i in self.indices]
    else:
      y = tf.gather(inputs, indices=self.indices, **self.kwargs)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'indices': self.indices,
      'validate_indices': self.validate_indices,
      'axis': self.axis,
      'batch_dims': self.batch_dims
    }
    base_config = super(Gather, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GatheredAdd(tf_layers.Add):

  # Initialization
  # ===================================
  def __init__(
    self,
    gathered=None,
    **kwargs
  ):
    super(GatheredAdd, self).__init__(**kwargs)
    if ((gathered is not None) and (not isinstance(gathered, (list, tuple)))):
      utils.raise_err(
        ValueError,
        "The inputs provided for 'gathered' must be a list or tuple, where " \
        "each element operates on a specific input tensor intended for the " \
        "addition operation. Each element should be a dictionary specifying " \
        "the options for 'tf.gather' (e.g., {'indices': [0], 'axis': -1}). " \
        "This ensures the correct extraction of tensor entries for the " \
        "addition operation."
      )
    self.gathered = gathered

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    if (self.gathered is not None):
      y = []
      for (i, x) in enumerate(inputs):
        if (self.gathered[i] is not None):
          y.append(tf.gather(x, **self.gathered[i]))
        else:
          y.append(x)
    else:
      y = inputs
    return super(GatheredAdd, self).call(y)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'gathered': self.gathered
    }
    base_config = super(GatheredAdd, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Eigen(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    use_pinv=False,
    inv_scaling=None,
    **kwargs
  ):
    super(Eigen, self).__init__(**kwargs)
    self.use_pinv = use_pinv
    self.inv_scaling = inv_scaling

  # Properties
  # ===================================
  @property
  def v(self):
    return self._v

  @v.setter
  def v(self, value):
    self._v = value

  @property
  def vinv(self):
    return self._vinv

  @vinv.setter
  def vinv(self, value=None):
    if (value is None):
      _v = self.v
      if (self.inv_scaling is not None):
        _v = _v * self.inv_scaling
      if self.use_pinv:
        _vinv = linalg.pinv(_v, rcond=1e-10)
      else:
        _vinv = tf.linalg.inv(_v)
      if (self.inv_scaling is not None):
        _vinv *= self.inv_scaling
      self._vinv = _vinv
    else:
      self._vinv = value

  # Calling
  # ===================================
  def call(self, inputs, training=None, mode="decompose"):
    if (mode == "decompose"):
      return self.decompose(inputs)
    else:
      return self.reconstruct(inputs)

  def decompose(self, x):
    l, v = tf.linalg.eig(x)
    self.v, self.vinv = v, None
    return tf.expand_dims(l, axis=-1)

  def reconstruct(self, x):
    # > f(x) = V @ diag(x) @ V^(-1)
    # > Shapes: v=[n,i,j], x=[n,j,1], vinv=[n,j,i]
    v = tf.cast(self.v, dtype=x.dtype)
    vinv = tf.cast(self.vinv, dtype=x.dtype)
    return tf.matmul(v, x*vinv)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pinv": self.use_pinv,
      "inv_scaling": self.inv_scaling
    }
    base_config = super(Eigen, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MatrixExp(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    use_pade=True,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None, # "float32" None
    **kwargs
  ):
    super(MatrixExp, self).__init__(**kwargs)
    self.use_pade = use_pade
    self.use_expmv = use_expmv
    self.max_degree = max_degree
    self.max_nb_squarings = max_nb_squarings
    self.expm_dtype = expm_dtype

  # Calling
  # ===================================
  def call(self, A, B=None, training=None):
    A, B = [self.recast(x, self.expm_dtype) for x in (A, B)]
    if (B is None):
      y = self.compute_expm(A)
    else:
      if self.use_expmv:
        y = linalg.expmv(A, B, deg=self.max_degree)
      else:
        y = tf.matmul(self.compute_expm(A), B)
    return self.recast(y, bkd.get_floatx())

  def recast(self, x, dtype):
    if (x is not None):
      if (dtype is not None):
        return tf.cast(x, dtype=dtype)
      else:
        return x

  def compute_expm(self, A):
    return linalg.expm(
      A=A,
      use_pade=self.use_pade,
      max_degree=self.max_degree,
      max_nb_squarings=self.max_nb_squarings
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pade": self.use_pade,
      "use_expmv": self.use_expmv,
      "max_degree": self.max_degree,
      "max_nb_squarings": self.max_nb_squarings,
      "expm_dtype": self.expm_dtype
    }
    base_config = super(MatrixExp, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinODE(Layer):
  """
  Integral solution of linear ODE with constant coefficients:
    > dydt = a @ y + b
    > y(t) = exp(a*t) @ y0 + (exp(a*t)-I) @ a^(-1) @ b
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    # Matrix exponential
    use_pade=True,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    # Non-invertibility loss
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(LinODE, self).__init__(**kwargs)
    # Matrix exponential layer
    self.use_pade = use_pade
    self.use_expmv = use_expmv
    self.max_degree = max_degree
    self.max_nb_squarings = max_nb_squarings
    self.mat_exp = MatrixExp(
      use_pade=use_pade,
      use_expmv=use_expmv,
      max_degree=max_degree,
      max_nb_squarings=max_nb_squarings
    )
    # Additional loss to avoid non-invertibility of the matrix
    self.add_inv_loss = add_inv_loss
    self.inv_loss_weight = inv_loss_weight

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Manipulate inputs
    t, y0, a, b = utils.unpack_list(inputs, nb_elem=4)
    t, y0, b = [tf.expand_dims(z, axis=-1) for z in (t, y0, b)]
    # Additional loss to avoid non-invertibility of the matrix
    if (training and self.add_inv_loss):
      self.compute_inv_loss(a)
    # Compute solution
    if training:
      y = tf.matmul(tf.linalg.inv(a), b)
    else:
      y = tf.linalg.solve(a, b)
    y = self.mat_exp(a*t, y0+y) - y
    return tf.squeeze(y, axis=-1)

  def compute_inv_loss(self, a):
    # Compute singular values
    s = tf.linalg.svd(a, compute_uv=False)
    # Compute condition number
    smin = tf.math.reduce_min(s, axis=-1) + bkd.get_epsilon()
    smax = tf.math.reduce_max(s, axis=-1)
    cond = smax / smin
    # Minimize condition number
    loss = tf.math.reduce_mean(tf.math.log(cond))
    # Add loss
    self.add_loss(self.inv_loss_weight * loss)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pade": self.use_pade,
      "use_expmv": self.use_expmv,
      "max_degree": self.max_degree,
      "max_nb_squarings": self.max_nb_squarings,
      "add_inv_loss": self.add_inv_loss,
      "inv_loss_weight": self.inv_loss_weight
    }
    base_config = super(LinODE, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class LinODE(Layer):
#   """
#   Integral solution of linear ODE with constant coefficients:
#     > dydt = a @ y + b
#     > y(t) = exp(a*t) @ y0 + (exp(a*t)-I) @ a^(-1) @ b
#   """

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     # Eigendecomposition
#     use_eig=False,
#     use_pinv=False,
#     real_only=False,
#     inv_scaling=1.0,
#     # Matrix exponential
#     use_pade=True,
#     use_expmv=False,
#     max_degree=None,
#     max_nb_squarings=None,
#     # Non-invertibility loss
#     add_inv_loss=False,
#     inv_loss_weight=1e-3,
#     **kwargs
#   ):
#     super(LinODE, self).__init__(**kwargs)
#     self.use_eig = use_eig
#     self.use_pinv = use_pinv
#     self.real_only = real_only
#     self.inv_scaling = inv_scaling
#     self.use_pade = use_pade
#     self.use_expmv = use_expmv
#     self.max_degree = max_degree
#     self.max_nb_squarings = max_nb_squarings
#     if self.use_eig:
#       self.eigen = Eigen(
#         use_pinv=self.use_pinv,
#         inv_scaling=self.inv_scaling
#       )
#     else:
#       self.mat_exp = MatrixExp(
#         use_pade=self.use_pade,
#         use_expmv=self.use_expmv,
#         max_degree=self.max_degree,
#         max_nb_squarings=self.max_nb_squarings
#       )
#     # Additional loss to avoid non-invertibility of the matrix
#     self.add_inv_loss = add_inv_loss
#     self.inv_loss_weight = inv_loss_weight

#   # Calling
#   # ===================================
#   def call(self, inputs, training=None):
#     t, y0, a, b = utils.unpack_list(inputs, nb_elem=4)
#     # Reshape
#     t, y0, b = [tf.expand_dims(z, axis=-1) for z in (t, y0, b)]
#     # Additional loss to avoid non-invertibility of the matrix
#     self.compute_inv_loss(a)
#     # Compute solution
#     y = tf.linalg.solve(matrix=a, rhs=b)
#     y = self.mat_exp(a*t, y0+y) - y
#     return tf.squeeze(y, axis=-1)

#   # def compute_expa(self, t, a, eig=None, training=None):
#   #   t = tf.expand_dims(t, axis=-1)
#   #   if self.use_eig:
#   #     # Eigen-decomposition
#   #     if (eig is None):
#   #       l = self.eigen(a, training=training, mode="decompose")
#   #     else:
#   #       l, self.eigen.v = eig
#   #       self.eigen.vinv = None
#   #     if self.real_only:
#   #       l = tf.math.real(l)
#   #     s = tf.math.exp(t*l)
#   #     return self.eigen(s, training=training, mode="reconstruct")
#   #   else:
#   #     return self.expm(a*t, training=training)

#   # def compute_inv_loss(self, a):
#   #   if self.add_inv_loss:
#   #     ws, wb = self.inv_loss_weight
#   #     # Compute determinant
#   #     det = tf.math.abs(tf.linalg.det(a)) + bkd.get_epsilon()
#   #     # Too small determinant: avoiding singular matrix
#   #     loss = ws * tf.math.reduce_mean(tf.math.log(1.0/det+1.0))
#   #     # Too big determinant: avoiding numerical overflow
#   #     loss += wb * tf.math.reduce_mean(tf.math.log(det+1.0))
#   #     # Add loss
#   #     self.add_loss(loss)

#   def compute_inv_loss(self, a):
#     if self.add_inv_loss:
#       # Compute singular values
#       s = tf.linalg.svd(a, compute_uv=False)
#       # Compute condition number
#       smin = tf.math.reduce_min(s, axis=-1) + bkd.get_epsilon()
#       smax = tf.math.reduce_max(s, axis=-1)
#       cond = smax / smin
#       # Minimize condition number
#       loss = tf.math.reduce_mean(tf.math.log(cond))
#       # Add loss
#       self.add_loss(self.inv_loss_weight * loss)

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       "use_eig": self.use_eig,
#       "use_pinv": self.use_pinv,
#       "real_only": self.real_only,
#       "use_pade": self.use_pade,
#       "max_degree": self.max_degree,
#       "max_nb_squarings": self.max_nb_squarings,
#       "inv_scaling": self.inv_scaling
#     }
#     base_config = super(LinODE, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Reduction(Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    indices=None,
    reduction="sum",
    **kwargs
  ):
    super(Reduction, self).__init__(**kwargs)
    # Indices
    self.nb_elem = 1
    self.indices = indices
    if (self.indices is not None):
      self.indices = tf.constant(self.indices, dtype="int32")
      self.indices = tf.reshape(self.indices, shape=[-1])
      self.nb_elem = tf.math.reduce_max(self.indices)+1
    # Reduction operation
    self.reduction = str(reduction)
    self.unsorted_segment_reduce = {
      "max": tf.math.unsorted_segment_max,
      "mean": tf.math.unsorted_segment_mean,
      "min": tf.math.unsorted_segment_min,
      "prod": tf.math.unsorted_segment_prod,
      "sum": tf.math.unsorted_segment_sum
    }[self.reduction]
    self.reduce = {
      "max": tf.math.reduce_max,
      "mean": tf.math.reduce_mean,
      "min": tf.math.reduce_min,
      "prod": tf.math.reduce_prod,
      "sum": tf.math.reduce_sum
    }[self.reduction]

  # Calling
  # ===================================
  def call(self, x, axis=1, keepdims=True, training=None):
    if (self.indices is None):
      y = self.reduce(x, axis=axis, keepdims=keepdims)
    else:
      ndim = tf.rank(x)
      axes = tf.range(0, ndim, delta=1)
      # Transpose in
      perm = tf.roll(axes, shift=-axis, axis=0)
      y = tf.transpose(y, perm=perm)
      # Reduce
      y = self.unsorted_segment_reduce(
        data=y,
        segment_ids=self.indices,
        num_segments=self.nb_elem
      )
      # Transpose out
      perm = tf.roll(axes, shift=axis, axis=0)
      y = tf.transpose(y, perm=perm)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "indices": ops.to_numpy(self.indices),
      "reduction": self.reduction
    }
    base_config = super(Reduction, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
