import numpy as np
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
  loss="mse",
  axis=0,
  gathered=None,
  reduction="auto",
  std_scaler=None,
  **kwargs
):
  """Get Keras callable/readable object for the specific loss function
  identifier.

  Args:
    loss: A 'str'/'list'/'dict' of names/callables for the loss functions used.
    axis: The dimensions to reduce for the 'mean' operation.
    gathered: Same structure as 'loss' input where each element must be a
    dictionary specifying the options for 'tf.gather' (e.g., {'indices': [0],
    'axis': -1}). This ensures the correct extraction of tensor entries for
    the loss computation.
    reduction: Type of 'tf.keras.losses.Reduction' to apply to loss.

  Returns:
    A 'list'/'dict' of loss function callables.

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
  # Check 'axis' variable
  if (axis not in (0, -1, None)):
    utils.raise_err(
      ValueError,
      "'axis' for loss function is not interpretable. " \
      "Valid options are: [0, -1, None]."
    )
  kwargs["axis"] = axis
  # Get "reduction" variable
  kwargs["reduction"] = {
    "auto": losses_utils.ReductionV2.AUTO,
    "none": losses_utils.ReductionV2.NONE,
    "sum": losses_utils.ReductionV2.SUM,
    "sum_over_batch_size": losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
  }[reduction.lower()]
  # Scaler/Gathered
  objects = {}
  loss_struct = tf.nest.map_structure(lambda x: 1, loss)
  for (k, obj) in (("std_scaler", std_scaler), ("gathered", gathered)):
    if (obj is not None):
      struct = tf.nest.map_structure(lambda x: 1, obj)
      if (struct != loss_struct):
        utils.raise_err(
          ValueError,
          f"The '{k}' input must have the same structure as 'loss' input."
        )
      else:
        objects[k] = obj
    else:
      objects[k] = tf.nest.map_structure(lambda x: None, loss_struct)
  # Return losses
  return tf.nest.map_structure(
    lambda l, g, s: get_loss_fn(
      identifier=l, gathered=g, std_scaler=s, **kwargs
    ), loss, objects["gathered"], objects["std_scaler"]
  )

def get_loss_fn(
  identifier=None,
  axis=0,
  gathered=None,
  reduction="auto",
  std_scaler=None,
  **kwargs
):
  kwargs.update({
    "axis": axis,
    "gathered": gathered,
    "reduction": reduction,
    "std_scaler": std_scaler
  })
  if isinstance(identifier, str):
    if (identifier.lower() in _LOSS_IDS):
      return {
        "mae":   MeanAbsoluteError,
        "malpe": MeanAbsoluteLogarithmicPercentageError,
        "mape":  MeanAbsolutePercentageError,
        "mse":   MeanSquaredError,
        "msle":  MeanSquaredLogarithmicError,
        "gnlll": GaussianNLLLoss,
        "zero":  Zero
      }[identifier.lower()](**kwargs)
    else:
      return identifier
  elif callable(identifier):
    return identifier
  elif (identifier is None):
    return None
  else:
    utils.raise_err(
      ValueError,
      f"Could not interpret loss function identifier: '{identifier}'."
    )

# Custom loss functions
# =====================================
@keras_export("keras.losses.PCMLossFunctionWrapper")
class PCMLossFunctionWrapper(tf_losses.LossFunctionWrapper):

  def __init__(
    self,
    fn,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name=None,
    **kwargs
  ):
    super().__init__(
      fn=None,
      reduction=reduction,
      name=name,
      **kwargs
    )
    self.axis = axis
    self.gathered = gathered
    if isinstance(self.gathered, np.ndarray):
      self.gathered = self.gathered.tolist()
    self.std_scaler = std_scaler
    if isinstance(self.std_scaler, np.ndarray):
      self.std_scaler = self.std_scaler.tolist()
    self.fn = self._make_loss_fn(fn)

  def _make_loss_fn(self, loss_fn):
    def _loss_fn(y_true, y_pred):
      for xform in (self._gather, self._convert, self._scale):
        y_true, y_pred = xform(y_true, y_pred)
      loss = loss_fn(y_true, y_pred)
      return tf_bkd.mean(loss, axis=self.axis) if self.axis else loss
    return _loss_fn

  def _gather(self, y_true, y_pred):
    if (self.gathered is not None):
      return (tf.gather(y, **self.gathered) for y in (y_true, y_pred))
    else:
      return y_true, y_pred

  def _convert(self, y_true, y_pred):
    y_pred = ops.to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return y_true, y_pred

  def _scale(self, y_true, y_pred):
    if (self.std_scaler is not None):
      m = self.std_scaler["mean"]
      s = self.std_scaler["std"]
      y_true = (y_true - m) / s
      y_pred = (y_pred - m) / s
    return y_true, y_pred

# MAE
# -------------------------------------
@keras_export("keras.losses.MeanAbsoluteError")
class MeanAbsoluteError(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="mean_absolute_error",
    **kwargs
  ):
    super().__init__(
      fn=mean_absolute_error,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="MAE")
def mean_absolute_error(y_true, y_pred):
  return tf.abs(y_true - y_pred)

# MALPE
# -------------------------------------
@keras_export("keras.losses.MeanAbsoluteLogarithmicPercentageError")
class MeanAbsoluteLogarithmicPercentageError(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="mean_absolute_logarithmic_percentage_error",
    **kwargs
  ):
    super().__init__(
      fn=mean_absolute_logarithmic_percentage_error,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="MALPE")
def mean_absolute_logarithmic_percentage_error(y_true, y_pred):
  y_pred = tf.math.log(tf_bkd.maximum(y_pred, bkd.get_epsilon()))
  y_true = tf.math.log(tf_bkd.maximum(y_true, bkd.get_epsilon()))
  return 100 * tf.abs((y_true - y_pred) / y_true)

# MAPE
# -------------------------------------
@keras_export("keras.losses.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="mean_absolute_percentage_error",
    **kwargs
  ):
    super().__init__(
      fn=mean_absolute_percentage_error,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="MAPE")
def mean_absolute_percentage_error(y_true, y_pred):
  loss = tf_bkd.maximum(tf.abs(y_true), bkd.get_epsilon())
  return 100 * tf.abs((y_true - y_pred) / loss)

# MSE
# -------------------------------------
@keras_export("keras.losses.MeanSquaredError")
class MeanSquaredError(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="mean_squared_error",
    **kwargs
  ):
    super().__init__(
      fn=mean_squared_error,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="MSE")
def mean_squared_error(y_true, y_pred):
  return tf.math.squared_difference(y_pred, y_true)

# MSLE
# -------------------------------------
@keras_export("keras.losses.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="mean_squared_logarithmic_error",
    **kwargs
  ):
    super().__init__(
      fn=mean_squared_logarithmic_error,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="MSLE")
def mean_squared_logarithmic_error(y_true, y_pred):
  y_pred = tf.math.log(tf_bkd.maximum(y_pred, bkd.get_epsilon()))
  y_true = tf.math.log(tf_bkd.maximum(y_true, bkd.get_epsilon()))
  return tf.math.squared_difference(y_pred, y_true)

# GNLLL
# -------------------------------------
@keras_export("keras.losses.GaussianNLLLoss")
class GaussianNLLLoss(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="gaussian_nll_loss",
    **kwargs
  ):
    super().__init__(
      fn=gaussian_nll_loss,
      axis=axis,
      gathered=gathered,
      std_scaler=None,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="GNLLL")
def gaussian_nll_loss(y_true, y_pred, full=False):
  # Get mean and variance
  y_pred_mu, y_pred_sigma = tf.split(y_pred, 2, axis=-1)
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
  return loss

# ZERO
# -------------------------------------
@keras_export("keras.losses.Zero")
class Zero(PCMLossFunctionWrapper):

  def __init__(
    self,
    axis=-1,
    gathered=None,
    std_scaler=None,
    reduction=losses_utils.ReductionV2.AUTO,
    name="zero",
    **kwargs
  ):
    super().__init__(
      fn=zero,
      axis=axis,
      gathered=gathered,
      std_scaler=std_scaler,
      reduction=reduction,
      name=name,
      **kwargs
    )

@tf.keras.utils.register_keras_serializable(package=_CFG.name, name="ZERO")
def zero(y_true, y_pred):
  return tf.constant(0.0, dtype=bkd.get_floatx())
