from pycomet import utils
from . import losses as losses_mod
from tensorflow.python.keras import metrics as met


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
      return losses_mod.get(identifier)
  elif callable(identifier):
    return identifier
  else:
    utils.raise_err(
      ValueError,
      f"Could not interpret metric function identifier: `{identifier}`."
    )
