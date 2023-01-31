import tensorflow as tf
import tensorflow.python.keras.losses as tf_losses

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend
from pycomet import config as cfg
from pycomet import utils


def get(
    loss='mse',
    axis=0,
    reduction='auto'
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


def get_loss_fn(
    identifier=None,
    axis=0,
    reduction='auto',
    *args,
    **kwargs
):
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
        "mae":   MeanAbsoluteError(axis, reduction, *args, **kwargs),
        "malpe": MeanAbsoluteLogarithmicPercentageError(axis, reduction, *args, **kwargs),
        "mape":  MeanAbsolutePercentageError(axis, reduction, *args, **kwargs),
        "mse":   MeanSquaredError(axis, reduction, *args, **kwargs),
        "msle":  MeanSquaredLogarithmicError(axis, reduction, *args, **kwargs),
        "gnlll": GaussianNLLLoss(axis, reduction, *args, **kwargs),
        "zero":  Zero(reduction, *args, **kwargs)
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
        loss_fn = mean_absolute_error(axis)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MAE')
def mean_absolute_error(axis=0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.abs(y_true - y_pred)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fn

# MALPE -----------------------------------------------------------------------
@keras_export('keras.losses.MeanAbsoluteLogarithmicPercentageError')
class MeanAbsoluteLogarithmicPercentageError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_absolute_logarithmic_percentage_error'
    ):
        loss_fn = mean_absolute_logarithmic_percentage_error(axis)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MALPE')
def mean_absolute_logarithmic_percentage_error(axis=0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(backend.maximum(y_pred, backend.epsilon()))
        y_true = tf.math.log(backend.maximum(y_true, backend.epsilon()))
        loss = 100. * tf.abs((y_true - y_pred) / y_true)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fn

# MAPE ------------------------------------------------------------------------
@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_absolute_percentage_error'
    ):
        loss_fn = mean_absolute_percentage_error(axis)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MAPE')
def mean_absolute_percentage_error(axis=0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        den = backend.maximum(tf.abs(y_true), backend.epsilon())
        loss = 100. * tf.abs((y_true - y_pred) / den)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fn

# MSE -------------------------------------------------------------------------
@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_squared_error'
    ):
        loss_fn = mean_squared_error(axis)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MSE')
def mean_squared_error(axis=0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.math.squared_difference(y_pred, y_true)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fn

# MSLE ------------------------------------------------------------------------
@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        name='mean_squared_logarithmic_error'
    ):
        loss_fn = mean_squared_logarithmic_error(axis)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='MSLE')
def mean_squared_logarithmic_error(axis=0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(backend.maximum(y_pred, backend.epsilon()))
        y_true = tf.math.log(backend.maximum(y_true, backend.epsilon()))
        loss = tf.math.squared_difference(y_pred, y_true)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fn

# GNLLL -----------------------------------------------------------------------
@keras_export('keras.losses.GaussianNLLLoss')
class GaussianNLLLoss(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        axis,
        reduction,
        full=False,
        name='gaussian_nll_loss'
    ):
        loss_fn = gaussian_nll_loss(axis, full)
        super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='GNLLL')
def gaussian_nll_loss(axis=0, full=False):
    def loss_fn(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # Get mean and variance
        y_pred_mu, y_pred_sigma = tf.split(y_pred, 2, axis=1)
        # Entries of variance must be non-negative
        if tf.reduce_any(y_pred_sigma < 0):
            utils.raise_value_err("Variance has negative entry/entries.")
        # Clamp variance for stability
        y_pred_sigma = backend.maximum(y_pred_sigma, backend.epsilon())
        # Calculate the error
        loss = tf.math.squared_difference(y_pred_mu, y_true)
        loss = 0.5 * (tf.math.log(y_pred_sigma) + loss**2 / y_pred_sigma)
        if full:
            loss += 0.5 * tf.math.log(2 * tf.math.pi)
        return backend.mean(loss, axis=axis) if axis else loss
    return loss_fns

# ZERO ------------------------------------------------------------------------
@keras_export('keras.losses.Zero')
class Zero(tf_losses.LossFunctionWrapper):
    def __init__(
        self,
        reduction,
        name='zero'
    ):
        super().__init__(zero, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=cfg._NAME, name='ZERO')
def zero():
    return tf.constant(0.0, dtype=backend.floatx())
