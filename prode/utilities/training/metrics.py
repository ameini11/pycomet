import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as met

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from prode.utilities.training import losses as losses_module
from prode.utilities import utils


# Custom metric functions
###########################################################################
@tf.keras.utils.register_keras_serializable(package='PrODE', name='MRE')
def mre(order, axis=0, elementwise=False):
    def mean_rel_error(y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        if elementwise:
            err = norm((y_pred - y_true) / y_true, order)
            err = tf.reduce_mean(err, axis=axis)
        else:
            err = tf.norm(y_pred - y_true, ord=order, axis=axis)
            err = err / tf.norm(y_true, ord=order, axis=axis)
        return err
    return mean_rel_error

def norm(tensor, order):
    tensor = math_ops.abs(tensor)
    if order == 1:
        return tensor
    else:
        return math_ops.pow(math_ops.pow(tensor, order), 1.0 / order)

# Utils
###########################################################################
def get(identifier):
    metric_identifier = {
        "accuracy": met.Accuracy(),
        "mean":     met.Mean(),
        "mae":      met.MeanAbsoluteError(),
        "mape":     met.MeanAbsolutePercentageError(),
        "mse":      met.MeanSquaredError(),
        "msle":     met.MeanSquaredLogarithmicError(),
        "rmse":     met.RootMeanSquaredError()
    }

    if isinstance(identifier, str):
        try:
            return metric_identifier[identifier]
        except:
            return losses_module.get(identifier)
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err(
            "Could not interpret metric function identifier: `{}`".format(
                identifier
            )
        )
