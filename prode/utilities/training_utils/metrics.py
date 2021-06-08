import numpy as np
import tensorflow.keras.metrics as met

from prode.utilities.training_utils import losses as losses_module
from prode.utilities import utils


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
        utils.raise_value_err("Could not interpret metric function identifier:", identifier)
