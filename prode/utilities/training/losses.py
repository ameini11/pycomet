import tensorflow as tf
import tensorflow.keras.losses as lss

from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.framework import ops
from prode.utilities import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='MAE')
def mae(axis):
    def mean_absolute_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        err    = math_ops.abs(y_pred - y_true)
        return tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_absolute_error

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MALPE')
def malpe(axis):
    def mean_absolute_log_percentage_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_pred = math_ops.log(K.maximum(y_pred, K.epsilon()))
        y_true = math_ops.log(K.maximum(y_true, K.epsilon()))
        err    = math_ops.abs((y_true - y_pred) / y_true)
        return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_absolute_log_percentage_error

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MAPE')
def mape(axis):
    def mean_absolute_percentage_error(y_true, y_pred):
        # Warning: see MeanSquaredError above.
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        err    = math_ops.abs((y_true - y_pred) \
                    / K.maximum(math_ops.abs(y_true), K.epsilon()))
        return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_absolute_percentage_error

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSE')
def mse(axis):
    def mean_squared_error(y_true, y_pred):
        # Warning: see MeanSquaredError above.
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        err    = math_ops.squared_difference(y_pred, y_true)
        return tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_squared_error

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSLE')
def msle(axis):
    def mean_squared_logarithmic_error(y_true, y_pred):
        # Warning: see MeanSquaredError above.
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_pred = math_ops.log(K.maximum(y_pred, K.epsilon()))
        y_true = math_ops.log(K.maximum(y_true, K.epsilon()))
        err    = math_ops.squared_difference(y_pred, y_true)
        return tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_squared_logarithmic_error

@tf.keras.utils.register_keras_serializable(package='PrODE', name='ZERO')
def zero(*args):
    return tf.constant(0., dtype=K.floatx())


def get(identifier, axis=-1):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))
    loss_identifier = {
        "mae":   mae(axis),
        "malpe": malpe(axis),
        "mape":  mape(axis),
        "mse":   mse(axis),
        "msle":  msle(axis),
        "zero":  zero,
    }
    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    elif identifier == None:
        return None
    else:
        utils.raise_value_err("Could not interpret loss function identifier: `{}`".format(identifier))
