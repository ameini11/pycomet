import tensorflow.python.keras.metrics as tf_met

from pycomet import utils
from pycomet.optimization import losses as losses_module


def get(identifier):
    metric_identifier = {
        "accuracy": tf_met.Accuracy(),
        "mean":     tf_met.Mean(),
        "mae":      tf_met.MeanAbsoluteError(),
        "mape":     tf_met.MeanAbsolutePercentageError(),
        "mse":      tf_met.MeanSquaredError(),
        "msle":     tf_met.MeanSquaredLogarithmicError(),
        "rmse":     tf_met.RootMeanSquaredError()
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
            f"Could not interpret metric function identifier: `{identifier}`."
        )
