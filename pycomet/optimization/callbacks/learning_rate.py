import numpy as np
import tensorflow as tf

from pycomet import utils
from keras.utils import io_utils
from tensorflow.keras import optimizers as tf_opt
from tensorflow.python.keras import callbacks as cb
from tensorflow.python.keras import backend as tf_bkd
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.platform import tf_logging as logging


@keras_export('keras.callbacks.LRPiecewiseConstantDecay')
class LRPiecewiseConstantDecay(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    boundaries,
    values,
    verbose=0
  ):
    super(LRPiecewiseConstantDecay, self).__init__()
    self.name = utils.get_class_name(self)
    if (len(boundaries) != len(values) - 1):
      utils.raise_err(
        ValueError,
        f"In '{self.name}' callback, the length of " \
        "'boundaries' should be 1 less than the length of 'values'."
      )
    self.boundaries = np.array(boundaries)
    self.values = np.array(values)
    self.verbose = verbose
    self.idx = 0

  # Callback methods
  # ===================================
  def on_epoch_end(self, epoch, logs=None):
    if (not hasattr(self.model.optimizer, "lr")):
      utils.raise_err(ValueError, "Optimizer must have a 'lr' attribute.")
    greater = (self.boundaries < epoch)
    if (True in greater):
      self.idx = np.where(greater)[0][-1] + 1
      new_lr = self.values[self.idx]
      tf_bkd.set_value(self.model.optimizer.lr, new_lr)
      if (self.verbose > 0):
        io_utils.print_msg(
          "\nEpoch %05d: LRPiecewiseConstantDecay reducing "
          "learning rate to %s." % (epoch + 1, new_lr)
        )

@keras_export('keras.callbacks.LearningRateTracker')
class LearningRateTracker(cb.CSVLogger):

  # Initialization
  # ===================================
  def __init__(
    self,
    log_dir=None,
    filename='',
    verbose=1
  ):
    super(LearningRateTracker, self).__init__(filename)
    self.log_dir = log_dir
    self.verbose = verbose

  # Callback methods
  # ===================================
  def set_model(self, model):
    super(LearningRateTracker, self).set_model(model)
    self._lr_writer = None
    if (self.log_dir is not None):
      _logdir = self.log_dir + '/scalars/lr'
      self._lr_writer = tf.summary.create_file_writer(_logdir)

  def on_epoch_begin(self, epoch, logs={}):
    # Get learning rate value
    lr_value = self.model.optimizer._decayed_lr(tf.float32)
    lr_value = float(tf_bkd.get_value(lr_value))
    # Write to TensorBoard
    if (self._lr_writer is not None):
      with self._lr_writer.as_default():
        tf.summary.scalar(
          name='epoch_learning_rate', data=lr_value, step=epoch
        )
    # Update logs
    logs = {"lr": lr_value}
    super(LearningRateTracker, self).on_epoch_end(epoch, logs)
    # Write to sreeen
    if (self.verbose > 0):
      io_utils.print_msg("Learning rate value = %.4e" % lr_value)

  def on_epoch_end(self, epoch, logs=None):
    pass

@keras_export('keras.callbacks.ReduceLROnPlateau')
class ReduceLROnPlateau(cb.ReduceLROnPlateau):

  # Initialization
  # ===================================
  def __init__(
    self,
    monitor='val_loss',
    factor=0.5,
    patience=200,
    verbose=1,
    mode='auto',
    min_delta=1e-4,
    cooldown=0,
    min_lr=1.e-6,
    **kwargs
  ):
    super(ReduceLROnPlateau, self).__init__(
      monitor=monitor,
      factor=factor,
      patience=patience,
      verbose=verbose,
      mode=mode,
      min_delta=min_delta,
      cooldown=cooldown,
      min_lr=min_lr,
      **kwargs
    )

  # Callback methods
  # ===================================
  def on_train_begin(self, logs=None):
    self._reset()
    self.is_lr_scheduler = isinstance(
      self.model.optimizer.lr, tf_opt.schedules.LearningRateSchedule
    )
    c1 = self.is_lr_scheduler
    c2 = (not hasattr(self.model.optimizer.lr, 'initial_learning_rate'))
    if (c1 and c2):
      utils.raise_err(
        ValueError,
        "'ReduceLROnPlateau' callback can be used together " \
        "with a learning rate scheduler only if the latter has " \
        "'initial_learning_rate' as attribute."
      )

  def on_epoch_end(self, epoch, logs={}):
    logs['lr'] = self._get_lr()
    current = logs.get(self.monitor)
    if (current is None):
      logging.warning(
        "Learning rate reduction is conditioned on metric '%s' " \
        "which is not available. Available metrics are: %s",
        self.monitor, ','.join(list(logs.keys()))
      )
    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0
      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif (not self.in_cooldown()):
        self.wait += 1
        if (self.wait >= self.patience):
          old_lr = logs['lr']
          if (old_lr > self.min_lr):
            new_lr = self._set_lr(old_lr)
            if (self.verbose > 0):
              io_utils.print_msg(
                '\nEpoch %05d: ReduceLROnPlateau reducing learning'
                ' rate to %s.' % (epoch + 1, new_lr)
              )
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def _get_lr(self):
    if self.is_lr_scheduler:
      lr_value = self.model.optimizer._decayed_lr(tf.float32)
    else:
      lr_value = self.model.optimizer.lr
    return float(tf_bkd.get_value(lr_value))

  def _set_lr(self, old_lr):
    if self.is_lr_scheduler:
      new_lr = self.model.optimizer.lr.initial_learning_rate
      new_lr = max(new_lr * self.factor, self.min_lr)
      self.model.optimizer.lr.initial_learning_rate = new_lr
    else:
      new_lr = old_lr * self.factor
      new_lr = max(new_lr, self.min_lr)
      tf_bkd.set_value(self.model.optimizer.lr, new_lr)
    return float(new_lr)
