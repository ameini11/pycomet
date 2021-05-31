import tensorflow as tf
import tensorflow.keras.losses as lss

from tensorflow.python.ops          import math_ops, array_ops
from tensorflow.python.keras        import backend as K
from tensorflow.python.keras.utils  import losses_utils
from tensorflow.python.framework    import ops

from . import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='MALPE')
def mean_absolute_log_percentage_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_pred = math_ops.log(K.maximum(y_pred, K.epsilon()))
    y_true = math_ops.log(K.maximum(y_true, K.epsilon()))
    err    = math_ops.abs((y_true - y_pred) / y_true)
    return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=-1))

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MAPE')
def mean_absolute_percentage_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    err    = math_ops.abs((y_true - y_pred) \
                / K.maximum(math_ops.abs(y_true), K.epsilon()))
    return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=0))

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSE')
def mean_squared_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    err    = math_ops.squared_difference(y_pred, y_true)
    return tf.reduce_sum(tf.reduce_mean(err, axis=0))

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSLE')
def mean_squared_logarithmic_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_pred = math_ops.log(K.maximum(y_pred, K.epsilon()))
    y_true = math_ops.log(K.maximum(y_true, K.epsilon()))
    err    = math_ops.squared_difference(y_pred, y_true)
    return tf.reduce_sum(tf.reduce_mean(err, axis=0))

@tf.keras.utils.register_keras_serializable(package='PrODE', name='ZERO')
def zero(*args):
    return tf.constant(0., dtype=K.floatx())


def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))
    loss_identifier = {
        "mae":   lss.MeanAbsoluteError(),
        "mape":  lss.MeanAbsolutePercentageError(),
        "mse":   lss.MeanSquaredError(),
        "msle":  lss.MeanSquaredLogarithmicError(),
        "MALPE": mean_absolute_log_percentage_error,
        "MAPE":  mean_absolute_percentage_error,
        "MSE":   mean_squared_error,
        "MSLE":  mean_squared_logarithmic_error,
        "ZERO":  zero,
    }
    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    elif identifier == None:
        return None
    else:
        utils.raise_value_err("Could not interpret loss function identifier:", identifier)


def get_reg_losses(model):
    regularization_losses = model.losses
    if regularization_losses:
        regularization_losses = losses_utils.cast_losses_to_common_dtype(
            regularization_losses)
        return math_ops.add_n(regularization_losses)
    else:
        return array_ops.zeros(shape=())