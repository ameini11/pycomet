import numpy as np
import tensorflow.keras.metrics as met

from . import losses as losses_module


def get(identifier):
    metric_identifier = {
        "accuracy": met.Accuracy(),
        "mean":     met.Mean(),
        "mae":      met.MeanAbsoluteError(),
        "mape":     met.MeanAbsolutePercentageError(),
        "mse":      met.MeanSquaredError(),
        "msle":     met.MeanSquaredLogarithmicError(),
        "rmse":     met.RootMeanSquaredError(),
        "MAPE":     losses_module.mean_absolute_percentage_error,
        "MALPE":    losses_module.mean_absolute_log_percentage_error,
        "MSE":      losses_module.mean_squared_error,
        "MSLE":     losses_module.mean_squared_logarithmic_error,
        "MSLPE":    losses_module.mean_squared_log_percentage_error
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret metric function identifier:", identifier)
