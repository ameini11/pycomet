from . import config
import tensorflow as tf
import tensorflow.keras.losses as lss


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    return tf.reduce_mean(tf.math.square(y_true - y_pred))


def zero(*_):
    return tf.constant(0, dtype=config.real(tf))


def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mean absolute error":              lss.MeanAbsoluteError(),
        "MAE":                              lss.MeanAbsoluteError(),
        "mae":                              lss.MeanAbsoluteError(),
        "mean squared error":               mean_squared_error,
        "MSE":                              mean_squared_error,
        "mse":                              mean_squared_error,
        "mean absolute percentage error":   lss.MeanAbsolutePercentageError(),
        "MAPE":                             lss.MeanAbsolutePercentageError(),
        "mape":                             lss.MeanAbsolutePercentageError(),
        "softmax cross entropy":            lss.CategoricalCrossentropy(),
        "zero":                             zero,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret loss function identifier:", identifier)
