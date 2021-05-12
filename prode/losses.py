from . import config
import tensorflow as tf
import tensorflow.keras.losses as lss


@tf.keras.utils.register_keras_serializable(package='prode', name='MSE')
def MeanSquaredError(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    err = tf.square(y_true - y_pred)
    return tf.reduce_sum(tf.reduce_mean(err, axis=0))


@tf.keras.utils.register_keras_serializable(package='prode', name='MAPE')
def MeanAbsolutePercentageError(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    err = 100.*tf.math.abs(tf.math.divide((y_true - y_pred), y_true))
    return tf.reduce_sum(tf.reduce_mean(err, axis=0))


@tf.keras.utils.register_keras_serializable(package='prode', name='MALPE')
def MeanAbsoluteLogPercentageError(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = 100.*tf.math.abs(tf.math.divide((y_true - y_pred), y_true))
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='MSLPE')
def MeanSquaredLogPercentageError(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = 100.*tf.square(tf.math.divide((y_true - y_pred), y_true))
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='MSLE')
def MeanSquaredLogarithmicError(y_true, y_pred):
    # Warning: see MeanSquaredError above.
    y_true = tf.math.log(y_true + 1.e-30)
    y_pred = tf.math.log(y_pred + 1.e-30)
    err    = tf.square(y_true - y_pred)
    return tf.reduce_sum(tf.reduce_mean(err, axis=-1))


@tf.keras.utils.register_keras_serializable(package='prode', name='ZERO')
def Zero(*_):
    return tf.constant(0., dtype=config.real(tf))


def get(identifier):
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    loss_identifier = {
        "mae":   lss.MeanAbsoluteError(),
        "mape":  lss.MeanAbsolutePercentageError(),
        "MAPE":  MeanAbsolutePercentageError,
        "MALPE": MeanAbsoluteLogPercentageError,
        "mse":   lss.MeanSquaredError(),
        "MSE":   MeanSquaredError,
        "msle":  lss.MSLE(),
        "MSLE":  MeanSquaredLogarithmicError,
        "MSLPE": MeanSquaredLogPercentageError,
        "ZERO":  Zero,
    }

    if isinstance(identifier, str):
        return loss_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret loss function identifier:", identifier)
