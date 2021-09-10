import tensorflow as tf
import tensorflow.keras.losses as lss

from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.framework import ops
from prode.utilities import utils


# Custom loss functions
###########################################################################
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


# Utils
###########################################################################
def get(identifier, axis=-1):
    """Get Keras callable/readable object for the specific loss function
    identifier.

    Args:
        identifier: Name/callable of the loss function.
        axis: Data matrix reduction axis for mean loss evaluation.

    Returns:
        A `list`/`dict` of loss function callables.

    Examples to be used in the input file:
        - Single output:
            loss = 'mae'
        - Single output + reduction axis:
            loss = {'identifier': 'mae', 'axis': 0}
        - Multi outputs:
            loss = ['mae', 'mape']
        - Multi outputs + reduction axis:
            loss = [{'identifier': 'mae', 'axis': 0},
                    {'identifier': 'mape', 'axis': 0}]

        >> If ODE data type are considered (see `model.get_loss_fn` first):
        - Single output:
            loss = {
                'ics': 'mae',
                'pts': 'mae',
                'res': 'mse',
            }
        - Single output + reduction axis:
            loss = {
                'ics': {'identifier': 'malpe', 'axis': 0},
                'pts': {'identifier': 'malpe', 'axis': 0},
                'res': {'identifier': 'mse', 'axis': 0},
            }
        - Multi outputs:
            loss = {
                'ics': ['mae', 'mape'],
                'pts': ['mae', 'mape'],
                'res': ['mse', 'mse']
                }
            }
        - Multi outputs + reduction axis:
            loss = {
                'ics': [{'identifier': 'malpe', 'axis': 0},
                        {'identifier': 'malpe', 'axis': 0}],
                'pts': [{'identifier': 'malpe', 'axis': 0},
                        {'identifier': 'malpe', 'axis': 0}],
                'res': [{'identifier': 'mse', 'axis': 0},
                        {'identifier': 'mse', 'axis': 0}]
                }
            }
        - Multi outputs (names included) + reduction axis:
            loss = {
                'ics': {
                    'out_1': {'identifier': 'malpe', 'axis': 0},
                    'out_2': {'identifier': 'malpe', 'axis': 0}
                },
                'pts': {
                    'out_1': {'identifier': 'malpe', 'axis': 0},
                    'out_2': {'identifier': 'malpe', 'axis': 0}
                },
                'res': {
                    'out_1': {'identifier': 'mse', 'axis': 0},
                    'out_2': {'identifier': 'mse', 'axis': 0}
                }
            }
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))
    
    if isinstance(identifier, dict):
        identifier, axis = identifier.values()

    loss_identifier = {
        "mae":   mae(axis),
        "malpe": malpe(axis),
        "mape":  mape(axis),
        "mse":   mse(axis),
        "msle":  msle(axis),
        "zero":  zero,
    }

    if isinstance(identifier, str):
        if identifier.lower() in loss_identifier:
            return loss_identifier[identifier.lower()]
        else:
            return identifier
    elif callable(identifier):
        return identifier
    elif identifier == None:
        return None
    else:
        utils.raise_value_err(
            "Could not interpret loss function identifier: `{}`".format(
                identifier
            )
        )
