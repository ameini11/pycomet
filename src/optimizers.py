from tensorflow.keras import optimizers as opt


def get_train_opt(optimizer, lr=None, decay=None):
    if lr is None:
        raise ValueError("No learning rate for {}.".format(optimizer))
    lr = _get_learningrate(lr, decay)
    return _get_optimizer(optimizer, lr)


def _get_learningrate(lr, decay):
    if decay is None:
        return lr, None
    return {
        "cosine":       opt.schedules.CosineDecay(lr, decay[1], alpha=decay[2]),
        "exponential":  opt.schedules.ExponentialDecay(lr, decay[1], decay[2], staircase=True),
        "inverse time": opt.schedules.InverseTimeDecay(lr, decay[1], decay[2], staircase=True),
        }[decay[0]]


def _get_optimizer(name, lr):
    return {
        "adadelta":     opt.Adadelta(learning_rate=lr),
        "adagrad":      opt.Adagrad(learning_rate=lr),
        "adam":         opt.Adam(learning_rate=lr),
        "adamax":       opt.Adamax(learning_rate=lr),
        "ftrl":         opt.Ftrl(learning_rate=lr),
        "nadam":        opt.Nadam(learning_rate=lr),
        "rmsprop":      opt.RMSprop(learning_rate=lr),
        "sgd":          opt.SGD(learning_rate=lr),
        "sgdnesterov":  opt.SGD(learning_rate=lr, nesterov=True),
    }[name]