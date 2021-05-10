import os
import sys
import time

import numpy as np
from datetime import datetime
# from .utils import save_animation

import tensorflow                   as tf
import tensorflow.keras.backend     as K
import tensorflow.keras.callbacks   as CB
from tensorflow.python.util.tf_export import keras_export

from .postprocessing import testing_plots


###############################################################################
### Learning Rate
###############################################################################
@keras_export('keras.callbacks.LRPerformanceDecay')
class LRPerformanceDecay(CB.ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced."""

    def __init__(
        self,
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=0,
        mode='auto',
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        **kwargs):
        super(ReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer._decayed_lr(tf.float64))
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))
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
                    old_lr = float(K.get_value(self.model.optimizer._decayed_lr(tf.float64)))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer._decayed_lr(tf.float64), new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                            'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


@keras_export('keras.callbacks.LRPiecewiseConstantDecay')
class LRPiecewiseConstantDecay(CB.Callback):
    """A Learning Rate Schedule that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries, values, verbose=0):
        super(LRPiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
            "The length of boundaries should be 1 less than the length of values")

        self.boundaries = np.array(boundaries)
        self.values     = np.array(values)
        self.idx        = 0

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        greater = self.boundaries < epoch
        if True in greater:
            idx = np.where(greater)[0][-1]
            self.idx = idx+1
            new_lr = self.values[self.idx]
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print('\nEpoch %05d: LRPiecewiseConstantDecay reducing learning '
                'rate to %s.' % (epoch + 1, new_lr))


@keras_export('keras.callbacks.LearningRateTracker')
class LearningRateTracker(CB.Callback):
    """Learning rate tracker."""

    def __init__(self, file=None, log_dir=None, verbose=True):
        super(LearningRateTracker, self).__init__()

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_lr = tf.summary.create_file_writer(log_dir + '/scalars/lr')

        self.file = None if file is None else open(file, "w", buffering=1)
        if self.file is not None: self.file.write("lr\n")

        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer._decayed_lr(tf.float64)))
        if self.log_dir is not None:
            with self.log_lr.as_default():
                tf.summary.scalar(name='epoch_lr', data=lr, step=epoch)
        if self.file is not None:
            self.file.write('%.6e\n' % lr)
            self.file.flush()
        if self.verbose > 0:
            print("Learning rate value = %.4e" % lr)

    def on_train_end(self, logs=None):
        if self.file is not None:
            self.file.close()


###############################################################################
### Time
###############################################################################
@keras_export('keras.callbacks.Timer')
class Timer(CB.Callback):
    """Stop training when training time reaches the threshold.
    This Timer starts after the first call of `on_train_begin`."""

    def __init__(self, available_time):
        super(Timer, self).__init__()

        self.threshold = available_time * 60  # convert to seconds
        self.t_start   = None

    def on_train_begin(self, logs=None):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self.t_start) > self.threshold:
            self.model.stop_training = True
            print("\nStop training as time used up.\n\
                   Time used: {:.1f} mins, epoch trained: {}".format(
                    (time.time() - self.t_start) / 60, epoch))


###############################################################################
### Variables
###############################################################################
@keras_export('keras.callbacks.VariableValue')
class VariableValue(CB.Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variables.
        period (int): Interval (number of epochs) between checking values.
        filepath (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
    """

    def __init__(self, var_list, period=1, filepath=None):
        super(VariableValue, self).__init__()

        self.var_list  = var_list
        self.names     = [weight.name for layer in self.model.layers for weight in layer.weights]
        self.idx       = [self.names.index(i) for i in self.var_list]

        self.period    = period
        self.precision = precision

        self.file = sys.stdout if filepath is None \
                    else open(filepath+'/var_values.csv', "w", buffering=1)
        if self.file is not sys.stdout: self.file.write("#epoch",*self.var_list, sep=',', end='\n')

        self.values = None
        self.epochs_since_last = 0

    def on_train_begin(self, logs=None):
        self.values = [list(self.model.get_weights())[i] for i in self.idx]
        if self.file is not sys.stdout:
            self.file.write("0",*self.values, sep=',',end='\n')
            self.file.flush()
        print("\nVariables values at epoch %05d:\n" % 0)
        for i, name in enumerate(self.var_list): print("  -", name, '=', self.values[i])

    def on_train_end(self, logs=None):
        if self.file is not sys.stdout:
            self.file.close()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.values = [list(self.model.get_weights())[i] for i in self.idx]
            if self.file is not sys.stdout:
                self.file.write(epoch,*self.values, sep=',',end='\n')
                self.file.flush()
            print("\nVariables values at epoch %05d:\n" % epoch)
            for i, name in enumerate(self.var_list): print("  -", name, '=', self.values[i])

    def get_value(self):
        """Return the variable values."""
        return self.values


###############################################################################
### Visualization
###############################################################################
class Plotter(CB.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        test_cases,
        var_names,
        labels,
        scales,
        log_flg,
        train_dir,
        period
    ):
        super(Plotter, self).__init__()
        self.test_cases = test_cases
        self.var_names = var_names
        self.labels = labels
        self.scales = scales
        self.log_flg = log_flg

        self.path = train_dir+'/testing_plots'
        if not os.path.exists(self.path): os.makedirs(self.path)

        self.period = period

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.period == 0:
            for i, test_i in enumerate(self.test_cases):
                if len(test_i) == 3:
                    title, x, y_true = test_i
                    data_val = None
                else:
                    title, x, y_true, data_val = test_i
                y_pred = self.model.predict(x)

                testing_plots.plot_test(self.path, x, y_true, y_pred, self.var_names,
                    title, i, labels=self.labels, scales=self.scales, data_val=data_val)
                if self.log_flg:
                    testing_plots.plot_test(self.path, x, y_true, y_pred, self.var_names,
                        title, i, labels=self.labels, scales=['log','log'], data_val=data_val)
               

# class MovieDumper(CB.Callback):
#     """Dump a movie to show the training progress of the function along a line.

#     Args:
#         spectrum: If True, dump the spectrum of the Fourier transform.
#     """

#     def __init__(
#         self,
#         filename,
#         x1,
#         x2,
#         num_points=100,
#         period=1,
#         component=0,
#         save_spectrum=False,
#         y_reference=None,
#     ):
#         super(MovieDumper, self).__init__()
#         self.filename = filename
#         x1 = np.array(x1)
#         x2 = np.array(x2)
#         self.x = x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]
#         self.period = period
#         self.component = component
#         self.save_spectrum = save_spectrum
#         self.y_reference = y_reference

#         self.y = []
#         self.spectrum = []
#         self.epochs_since_last_save = 0

#     def init(self):
#         self.tf_op = self.model.net.outputs[:, self.component]
#         self.feed_dict = self.model.net.feed_dict(False, False, 2, self.x)

#     def on_train_begin(self):
#         self.y.append(self.model.sess.run(self.tf_op, feed_dict=self.feed_dict))
#         if self.save_spectrum:
#             A = np.fft.rfft(self.y[-1])
#             self.spectrum.append(np.abs(A))

#     def on_epoch_end(self):
#         self.epochs_since_last_save += 1
#         if self.epochs_since_last_save >= self.period:
#             self.epochs_since_last_save = 0
#             self.on_train_begin()

#     def on_train_end(self):
#         fname_x = self.filename + "_x.txt"
#         fname_y = self.filename + "_y.txt"
#         fname_movie = self.filename + "_y.gif"
#         print(
#             "\nSaving the movie of function to {}, {}, {}...".format(
#                 fname_x, fname_y, fname_movie
#             )
#         )
#         np.savetxt(fname_x, self.x)
#         np.savetxt(fname_y, np.array(self.y))
#         if self.y_reference is None:
#             save_animation(fname_movie, np.ravel(self.x), self.y)
#         else:
#             y_reference = np.ravel(self.y_reference(self.x))
#             save_animation(
#                 fname_movie, np.ravel(self.x), self.y, y_reference=y_reference
#             )

#         if self.save_spectrum:
#             fname_spec = self.filename + "_spectrum.txt"
#             fname_movie = self.filename + "_spectrum.gif"
#             print(
#                 "Saving the movie of spectrum to {}, {}...".format(
#                     fname_spec, fname_movie
#                 )
#             )
#             np.savetxt(fname_spec, np.array(self.spectrum))
#             xdata = np.arange(len(self.spectrum[0]))
#             if self.y_reference is None:
#                 save_animation(fname_movie, xdata, self.spectrum, logy=True)
#             else:
#                 A = np.fft.rfft(y_reference)
#                 save_animation(
#                     fname_movie, xdata, self.spectrum, logy=True, y_reference=np.abs(A)
#                 )
