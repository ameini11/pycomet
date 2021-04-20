import numpy as np
import tensorflow.keras.metrics as met
from . import config


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
        return metric_identifier[identifier.lower()]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret metric function identifier:", identifier)
