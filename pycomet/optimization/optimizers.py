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
    "exponential": opt.schedules.ExponentialDecay,
    "inverse_time": opt.schedules.InverseTimeDecay,
    "polynomial": opt.schedules.PolynomialDecay
  }[name](**scheduler)

def get_optimizer(name, lr):
  return {
    "adadelta":     opt.Adadelta,
    "adagrad":      opt.Adagrad,
    "adam":         opt.Adam,
    "adamax":       opt.Adamax,
    "ftrl":         opt.Ftrl,
    "rmsprop":      opt.RMSprop,
    "sgd":          opt.SGD
  }[name](learning_rate=lr)
