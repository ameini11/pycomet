import os
import numpy as np
import pandas as pd
import tensorflow.keras.callbacks as CB

from tensorflow.python.util.tf_export import keras_export
from prode.postprocessing import testing_plots, training_plots


@keras_export('keras.callbacks.Plotter')
class Plotter(CB.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        test_cases,
        var_names,
        labels,
        scales,
        log_flg,
        limit_y,
        train_dir,
        period):
        super(Plotter, self).__init__()

        self.test_cases = test_cases
        self.var_names  = var_names
        self.labels     = labels
        self.scales     = scales
        self.log_flg    = log_flg

        self.limit_y    = limit_y

        self.train_dir  = train_dir
        self.path = {
            'train': self.train_dir+'plots/training',
            'test':  self.train_dir+'plots/testing'
        }
        for k, v in self.path.items():
            if not os.path.exists(v):
                os.makedirs(v)
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            print(">> `Plotter` callback:")
            self.training(limit_y=self.limit_y)
            self.testing()

    def training(self, scale='log', limit_y=False):
        print("   - Plotting training quantities ...")
        history = pd.DataFrame.from_dict(self.model.history.history)
        path = self.path['train']
        # Learning rate
        training_plots.plot_lr(self.train_dir+'/lr.csv', path)
        # Losses
        if 'loss' in history:
            training_plots.plot_loss(history, path, scale=scale, limit_y=limit_y)
        elif 'tot_loss' in history:
            training_plots.plot_composite_loss(history, path, scale=scale, limit_y=limit_y)
            if any([ c.endswith('weight') for c in history ]):
                training_plots.plot_loss_weights(history, path)
        # Metrics
        metrics = self.model.compiled_metrics._metrics
        if metrics:
            for metric in metrics:
                training_plots.plot_metric(history, path, metric._name, scale=scale, limit_y=limit_y)

    def testing(self):
        print("   - Plotting test cases ...")
        for i, test_i in enumerate(self.test_cases):
            title, x, y_true = self.unpack_test_data(test_i)
            y_pred = self.model.predict(x)

            path = self.path['test'] + '/test_'+str(i+1)
            if not os.path.exists(path):
                os.makedirs(path)
            testing_plots.plot_test(path, x, y_true, y_pred, self.var_names,
                title=title, labels=self.labels, scales=self.scales)
            if self.log_flg:
                testing_plots.plot_test(path, x, y_true, y_pred, self.var_names,
                    title=title, labels=self.labels, scales=['log','log'])

    def unpack_test_data(self, test_data):
        if len(test_data) == 3:
            title, x, y_true = test_data
        elif len(test_data) == 2:
            x, y_true = test_data
            title = None
        else:
            x = test_data
            title = y_true = None
        return title, x, y_true


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
