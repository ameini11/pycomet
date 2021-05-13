import tensorflow as tf
import tensorflow.keras.losses as lss

from tensorflow.python.keras import backend as K


@tf.keras.utils.register_keras_serializable(package='prode', name='MALPE')
def mean_absolute_log_percentage_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = 100.*tf.math.abs(tf.math.divide((y_true - y_pred), y_true))
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='MSLE')
def mean_squared_logarithmic_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = tf.square(y_true - y_pred)
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='MSLPE')
def mean_squared_log_percentage_error(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = 100.*tf.square(tf.math.divide((y_true - y_pred), y_true))
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='ZERO')
def zero(*_):
    return tf.constant(0., dtype=K.floatx())


def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mae":   lss.MeanAbsoluteError(),
        "mape":  lss.MeanAbsolutePercentageError(),
        "mse":   lss.MeanSquaredError(),
        "msle":  lss.MeanSquaredLogarithmicError(),
        "MALPE": mean_absolute_log_percentage_error,
        "MSLE":  mean_squared_logarithmic_error,
        "MSLPE": mean_squared_log_percentage_error,
        "zero":  zero,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret loss function identifier:", identifier)
