import tensorflow as tf
import tensorflow.python.keras.losses as tf_losses

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K
from prode.utilities import utils


def get(
    loss='mse',
    axis=None,
    reduction='sum_over_batch_size'
):
    # Check `axis` variable
    if axis not in {0, -1, None}:
        utils.raise_value_err(
            "`axis` for loss function is not interpretable. Valid "
            "options are: {0, -1, None}."
        )
    # Get `reduction` variable
    reduction = {
        'auto': losses_utils.ReductionV2.AUTO,
        'none': losses_utils.ReductionV2.NONE,
        'sum': losses_utils.ReductionV2.SUM,
        'sum_over_batch_size': losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    }[reduction.lower()]

    return tf.nest.map_structure(
        lambda l: get_loss_fn(l, axis=axis, reduction=reduction), loss
    )


def get_loss_fn(identifier=None, axis=-1, reduction=None):
    """Get Keras callable/readable object for the specific loss function
    identifier.

    Args:
        identifier: Name/callable of the loss function.
        axis: The dimensions to reduce for the `mean` operation.
        reduction: Type of `tf.keras.losses.Reduction` to apply to loss.

    Returns:
        A `list`/`dict` of loss function callables.

    Examples to be used in the input file:
        - Single output:
            loss = 'mae'
        - Multi outputs:
            loss = ['mae', 'mape']

        >> If multi-objective functions loss is considered:
        - Single output:
            loss = {
                'ics': 'mae',
                'pts': 'mae',
                'res': 'mse'
            }
        - Multi outputs:
            loss = {
                'ics': ['mae', 'mape'],
                'pts': ['mae', 'mape'],
                'res': ['mse', 'mse']
            }
    """
    loss_identifier = {
        "mae":   MeanAbsoluteError(axis, reduction),
        "malpe": MeanAbsoluteLogarithmicPercentageError(axis, reduction),
        "mape":  MeanAbsolutePercentageError(axis, reduction),
        "mse":   MeanSquaredError(axis, reduction),
        "msle":  MeanSquaredLogarithmicError(axis, reduction),
        "zero":  Zero(axis, reduction)
    }
    if isinstance(identifier, str):
        if identifier.lower() in loss_identifier:
            return loss_identifier[identifier.lower()]
        else:
            return identifier
    elif callable(identifier):
        return identifier
    elif identifier is None:
        return None
    else:
        utils.raise_value_err(
            f"Could not interpret loss function identifier: `{identifier}`."
        )


# Custom loss functions
###############################################################################
# MAE -------------------------------------------------------------------------
@keras_export('keras.losses.MeanAbsoluteError')
class MeanAbsoluteError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_absolute_error'
    ):
        err_fn = mean_absolute_error(axis)
        super().__init__(err_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MAE')
def mean_absolute_error(axis):
    def err_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        err = tf.abs(y_true - y_pred)
        return K.mean(err, axis=axis) if axis else err
    return err_fn

# MALPE -----------------------------------------------------------------------
@keras_export('keras.losses.MeanAbsoluteLogarithmicPercentageError')
class MeanAbsoluteLogarithmicPercentageError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_absolute_logarithmic_percentage_error'
    ):
        err_fn = mean_absolute_logarithmic_percentage_error(axis)
        super().__init__(err_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MALPE')
def mean_absolute_logarithmic_percentage_error(axis):
    def err_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(K.maximum(y_pred, K.epsilon()))
        y_true = tf.math.log(K.maximum(y_true, K.epsilon()))
        err = 100. * tf.abs((y_true - y_pred) / y_true)
        return K.mean(err, axis=axis) if axis else err
    return err_fn

# MAPE ------------------------------------------------------------------------
@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_absolute_percentage_error'
    ):
        err_fn = mean_absolute_percentage_error(axis)
        super().__init__(err_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MAPE')
def mean_absolute_percentage_error(axis):
    def err_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        err = 100. * tf.abs(
            (y_true - y_pred) / K.maximum(tf.abs(y_true), K.epsilon())
        )
        return K.mean(err, axis=axis) if axis else err
    return err_fn

# MSE -------------------------------------------------------------------------
@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_squared_error'
    ):
        print(axis)
        err_fn = mean_squared_error(axis)
        super().__init__(err_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSE')
def mean_squared_error(axis):
    def err_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        err = tf.math.squared_difference(y_pred, y_true)
        return K.mean(err, axis=axis) if axis else err
    return err_fn

# MSLE ------------------------------------------------------------------------
@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_squared_logarithmic_error'
    ):
        err_fn = mean_squared_logarithmic_error(axis)
        super().__init__(err_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='MSLE')
def mean_squared_logarithmic_error(axis):
    def err_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(K.maximum(y_pred, K.epsilon()))
        y_true = tf.math.log(K.maximum(y_true, K.epsilon()))
        err = tf.math.squared_difference(y_pred, y_true)
        return K.mean(err, axis=axis) if axis else err
    return err_fn

# ZERO ------------------------------------------------------------------------
@keras_export('keras.losses.Zero')
class Zero(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='zero'
    ):
        super().__init__(zero, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package='PrODE', name='ZERO')
def zero():
    return tf.constant(0., dtype=K.floatx())
