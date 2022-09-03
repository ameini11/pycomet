import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers as tf_opt
from tensorflow.python.keras import backend
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import callbacks as cb
from pycomet import utils


@keras_export('keras.callbacks.LRPiecewiseConstantDecay')
class LRPiecewiseConstantDecay(cb.Callback):
    """A Learning Rate Schedule that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries, values, verbose=0):
        super(LRPiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            utils.raise_value_err(
                f"In `{self.__name__}` callback, the length of "
                "`boundaries` should be 1 less than the length of `values`."
            )
        self.boundaries = np.array(boundaries)
        self.values = np.array(values)
        self.idx = 0

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            utils.raise_value_err("Optimizer must have a `lr` attribute.")
        greater = self.boundaries < epoch
        if True in greater:
            idx = np.where(greater)[0][-1]
            self.idx = idx+1
            new_lr = self.values[self.idx]
            backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                utils.print_fn(
                    "Epoch %05d: LRPiecewiseConstantDecay reducing "
                    "learning rate to %s." % (epoch + 1, new_lr), lev=0
                )


@keras_export('keras.callbacks.LearningRateTracker')
class LearningRateTracker(cb.CSVLogger):

    def __init__(
        self,
        log_dir=None,
        filename='',
        verbose=1
    ):
        super(LearningRateTracker, self).__init__(filename)

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_lr = tf.summary.create_file_writer(
                self.log_dir + '/scalars/lr'
            )

        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        logs = {'lr': float(backend.get_value(
            self.model.optimizer._decayed_lr(tf.float32)
        ))}
        if self.log_dir is not None:
            with self.log_lr.as_default():
                tf.summary.scalar(
                    name='epoch_learning_rate', data=logs['lr'], step=epoch
                )
        super(LearningRateTracker, self).on_epoch_end(epoch, logs)
        if self.verbose > 0:
            print("Learning rate value = %.4e" % logs['lr'])

    def on_epoch_end(self, epoch, logs=None):
        pass


@keras_export('keras.callbacks.ReduceLROnPlateau')
class ReduceLROnPlateau(cb.ReduceLROnPlateau):
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
        super(ReduceLROnPlateau, self).__init__()

    def on_train_begin(self, logs=None):
        self._reset()
        self.is_lr_scheduler = isinstance(
            self.model.optimizer.lr, tf_opt.schedules.LearningRateSchedule
        )
        if self.is_lr_scheduler and not hasattr(
            self.model.optimizer.lr, 'initial_learning_rate'
        ):
            utils.raise_value_err(
                "'ReduceLROnPlateau' callback can be used together "
                "with a learning rate scheduler only if the latter has "
                "'initial_learning_rate' as attribute."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.is_lr_scheduler:
            logs['lr'] = backend.get_value(
                self.model.optimizer._decayed_lr(tf.float32)
            )
        else:
            logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                'Learning rate reduction is conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor, ','.join(list(logs.keys()))
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    if self.is_lr_scheduler:
                        old_lr = backend.get_value(
                            self.model.optimizer._decayed_lr(tf.float64)
                        )
                    else:
                        old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr > self.min_lr:
                        if self.is_lr_scheduler:
                            new_lr = self.model.optimizer.lr.initial_learning_rate * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            self.model.optimizer.lr.initial_learning_rate = new_lr
                        else:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            backend.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: ReduceLROnPlateau reducing learning'
                                ' rate to %s.' % (epoch + 1, new_lr)
                            )  
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
