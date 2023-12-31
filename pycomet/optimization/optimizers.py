from pycomet import utils
from tensorflow.keras import optimizers as opt


def get_train_opt(optimizer, lr=None, scheduler=None):
  if (lr is None):
    utils.raise_err(ValueError, f"No learning rate for '{optimizer}'.")
  lr = get_learning_rate(lr, scheduler)
  return get_optimizer(optimizer, lr)

def get_learning_rate(lr, scheduler=None):
  if (scheduler is None):
    return lr
  name = scheduler.pop("name")
  scheduler["initial_learning_rate"] = lr
  return {
    "exponential": opt.schedules.ExponentialDecay(**scheduler),
    "inverse_time": opt.schedules.InverseTimeDecay(**scheduler),
    "polynomial": opt.schedules.PolynomialDecay(**scheduler)
  }[name]

def get_optimizer(name, lr):
  return {
    "adadelta":     opt.Adadelta(learning_rate=lr),
    "adagrad":      opt.Adagrad(learning_rate=lr),
    "adam":         opt.Adam(learning_rate=lr),
    "adamax":       opt.Adamax(learning_rate=lr),
    "ftrl":         opt.Ftrl(learning_rate=lr),
    "rmsprop":      opt.RMSprop(learning_rate=lr),
    "sgd":          opt.SGD(learning_rate=lr),
    "sgd_nesterov": opt.SGD(learning_rate=lr, nesterov=True)
  }[name]
