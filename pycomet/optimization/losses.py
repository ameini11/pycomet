import tensorflow as tf
import tensorflow.python.keras.losses as tf_losses

from tensorflow.python.keras import backend as tf_bkd
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export
from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
_CFG = get_config()


_LOSS_IDS = ("mae", "malpe", "mape", "mse", "msle", "gnlll", "zero")

def get(
  loss='mse',
  axis=0,
  reduction='auto',
  standard_scaler=None,
  **kwargs
):
  # Check `axis` variable
  if (axis not in (0, -1, None)):
    utils.raise_err(
      ValueError,
      "`axis` for loss function is not interpretable. Valid " \
      "options are: [0, -1, None]."
    )
  kwargs['axis'] = axis
  # Get `reduction` variable
  reduction = {
    'auto': losses_utils.ReductionV2.AUTO,
    'none': losses_utils.ReductionV2.NONE,
    'sum': losses_utils.ReductionV2.SUM,
    'sum_over_batch_size': losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
  }[reduction.lower()]
  kwargs['reduction'] = reduction
  # Scaler
  if (standard_scaler is not None):
    if (not isinstance(standard_scaler, dict)):
      utils.raise_err(
        ValueError,
        "`standard_scaler` for loss function has to " \
        "be a `dict` with 'mean' and 'std' as keys."
      )
  kwargs['standard_scaler'] = standard_scaler
  # Return losses
  return tf.nest.map_structure(lambda l: get_loss_fn(l, **kwargs), loss)

def get_loss_fn(
  identifier=None,
  axis=0,
  reduction='auto',
  standard_scaler=None,
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
  kwargs.update({
    'axis': axis,
    'reduction': reduction,
    'standard_scaler': standard_scaler
  })
  if isinstance(identifier, str):
    if (identifier.lower() in _LOSS_IDS):
      return {
        "mae":   MeanAbsoluteError(**kwargs),
        "malpe": MeanAbsoluteLogarithmicPercentageError(**kwargs),
        "mape":  MeanAbsolutePercentageError(**kwargs),
        "mse":   MeanSquaredError(**kwargs),
        "msle":  MeanSquaredLogarithmicError(**kwargs),
        "gnlll": GaussianNLLLoss(**kwargs),
        "zero":  Zero(**kwargs)
      }[identifier.lower()]
    else:
      return identifier
  elif callable(identifier):
    return identifier
  elif (identifier is None):
    return None
  else:
    utils.raise_err(
      ValueError,
      f"Could not interpret loss function identifier: `{identifier}`."
    )

# Custom loss functions
# =====================================
# MAE
# -------------------------------------
@keras_export('keras.losses.MeanAbsoluteError')
class MeanAbsoluteError(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    standard_scaler,
    name='mean_absolute_error',
    **kwargs
  ):
    loss_fn = mean_absolute_error(axis, standard_scaler)
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='MAE')
def mean_absolute_error(axis=0, standard_scaler=None):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    y_pred, y_true = _scale(y_pred, y_true, standard_scaler)
    loss = tf.abs(y_true - y_pred)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# MALPE 
# -------------------------------------
@keras_export('keras.losses.MeanAbsoluteLogarithmicPercentageError')
class MeanAbsoluteLogarithmicPercentageError(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    standard_scaler,
    name='mean_absolute_logarithmic_percentage_error',
    **kwargs
  ):
    loss_fn = mean_absolute_logarithmic_percentage_error(
      axis, standard_scaler
    )
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='MALPE')
def mean_absolute_logarithmic_percentage_error(axis=0, standard_scaler=None):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    y_pred, y_true = _scale(y_pred, y_true, standard_scaler)
    y_pred = tf.math.log(tf_bkd.maximum(y_pred, bkd.get_epsilon()))
    y_true = tf.math.log(tf_bkd.maximum(y_true, bkd.get_epsilon()))
    loss = 100 * tf.abs((y_true - y_pred) / y_true)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# MAPE 
# -------------------------------------
@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    standard_scaler,
    name='mean_absolute_percentage_error',
    **kwargs
  ):
    loss_fn = mean_absolute_percentage_error(axis, standard_scaler)
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='MAPE')
def mean_absolute_percentage_error(axis=0, standard_scaler=None):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    y_pred, y_true = _scale(y_pred, y_true, standard_scaler)
    den = tf_bkd.maximum(tf.abs(y_true), bkd.get_epsilon())
    loss = 100 * tf.abs((y_true - y_pred) / den)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# MSE 
# -------------------------------------
@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    standard_scaler,
    name='mean_squared_error',
    **kwargs
  ):
    loss_fn = mean_squared_error(axis, standard_scaler)
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='MSE')
def mean_squared_error(axis=0, standard_scaler=None):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    y_pred, y_true = _scale(y_pred, y_true, standard_scaler)
    loss = tf.math.squared_difference(y_pred, y_true)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# MSLE 
# -------------------------------------
@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    standard_scaler,
    name='mean_squared_logarithmic_error',
    **kwargs
  ):
    loss_fn = mean_squared_logarithmic_error(axis, standard_scaler)
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='MSLE')
def mean_squared_logarithmic_error(axis=0, standard_scaler=None):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    y_pred, y_true = _scale(y_pred, y_true, standard_scaler)
    y_pred = tf.math.log(tf_bkd.maximum(y_pred, bkd.get_epsilon()))
    y_true = tf.math.log(tf_bkd.maximum(y_true, bkd.get_epsilon()))
    loss = tf.math.squared_difference(y_pred, y_true)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# GNLLL 
# -------------------------------------
@keras_export('keras.losses.GaussianNLLLoss')
class GaussianNLLLoss(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    axis,
    reduction,
    full=False,
    name='gaussian_nll_loss',
    **kwargs
  ):
    loss_fn = gaussian_nll_loss(axis, full)
    super().__init__(loss_fn, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='GNLLL')
def gaussian_nll_loss(axis=0, full=False):
  def loss_fn(y_true, y_pred):
    y_pred, y_true = _convert(y_pred, y_true)
    # Get mean and variance
    y_pred_mu, y_pred_sigma = tf.split(y_pred, 2, axis=1)
    # Entries of variance must be non-negative
    if tf.reduce_any(y_pred_sigma < 0):
      utils.raise_err(ValueError, "Variance has negative entry/entries.")
    # Clamp variance for stability
    y_pred_sigma = tf.maximum(y_pred_sigma, bkd.get_epsilon())
    # Calculate the error
    loss = tf.math.squared_difference(y_pred_mu, y_true)
    loss = 0.5 * (tf.math.log(y_pred_sigma) + loss**2 / y_pred_sigma)
    if full:
      loss += 0.5 * tf.math.log(2 * tf.math.pi)
    return tf_bkd.mean(loss, axis=axis) if axis else loss
  return loss_fn

# Zero 
# -------------------------------------
@keras_export('keras.losses.Zero')
class Zero(tf_losses.LossFunctionWrapper):
  def __init__(
    self,
    reduction,
    name='zero',
    **kwargs
  ):
    super().__init__(zero, name=name, reduction=reduction)

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name='ZERO')
def zero():
  return tf.constant(0.0, dtype=bkd.get_floatx())

# Util functions
# -------------------------------------
def _convert(
  y_pred,
  y_true
):
  y_pred = ops.to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  return y_pred, y_true

def _scale(
  y_pred,
  y_true,
  standard_scaler=None
):
  if (standard_scaler is not None):
    m, s = standard_scaler['mean'], standard_scaler['std']
    y_pred = (y_pred - m) / s
    y_true = (y_true - m) / s
  return y_pred, y_true
