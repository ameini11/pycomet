from tensorflow.keras import optimizers as tf_opt
from prode.utilities import utils


def get_train_opt(optimizer, lr=None, decay=None):
    if lr is None:
        utils.raise_value_err("No learning rate for {}.".format(optimizer))
    lr = get_learning_rate(lr, decay)
    return get_optimizer(optimizer, lr)


def get_learning_rate(lr, decay):
    if decay is None:
        return lr
    return {
        "exponential": tf_opt.schedules.ExponentialDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "inverse_time": tf_opt.schedules.InverseTimeDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "polynomial": tf_opt.schedules.PolynomialDecay(
            lr, decay[1], end_learning_rate=1.e-2*lr
        )
    }[decay[0]]


def get_optimizer(name, lr):
    return {
        "adadelta":     tf_opt.Adadelta(learning_rate=lr),
        "adagrad":      tf_opt.Adagrad(learning_rate=lr),
        "adam":         tf_opt.Adam(learning_rate=lr),
        "adamax":       tf_opt.Adamax(learning_rate=lr),
        "ftrl":         tf_opt.Ftrl(learning_rate=lr),
        "rmsprop":      tf_opt.RMSprop(learning_rate=lr),
        "sgd":          tf_opt.SGD(learning_rate=lr),
        "sgd_nesterov": tf_opt.SGD(learning_rate=lr, nesterov=True)
    }[name]
