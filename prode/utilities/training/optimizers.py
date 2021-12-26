from tensorflow.keras import optimizers as opt
from prode.utilities import utils


def get_train_opt(optimizer, lr=None, decay=None):
    if lr is None:
        utils.raise_value_err("No learning rate for {}.".format(optimizer))
    lr = _get_learning_rate(lr, decay)
    return _get_optimizer(optimizer, lr)


def _get_learning_rate(lr, decay):
    if decay is None:
        return lr
    return {
        "exponential":  opt.schedules.ExponentialDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "inverse_time": opt.schedules.InverseTimeDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "polynomial":   opt.schedules.PolynomialDecay(
            lr, decay[1], end_learning_rate=1.e-2*lr
        )
    }[decay[0]]


def _get_optimizer(name, lr):
    return {
        "adadelta": opt.Adadelta(learning_rate=lr),
        "adagrad": opt.Adagrad(learning_rate=lr),
        "adam": opt.Adam(learning_rate=lr),
        "adamax": opt.Adamax(learning_rate=lr),
        "ftrl": opt.Ftrl(learning_rate=lr),
        "rmsprop": opt.RMSprop(learning_rate=lr),
        "sgd": opt.SGD(learning_rate=lr),
        "sgd_nesterov": opt.SGD(learning_rate=lr, nesterov=True)
    }[name]
