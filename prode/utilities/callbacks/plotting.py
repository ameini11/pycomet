import os
import copy
import numpy as np
import pandas as pd
import tensorflow.python.keras.callbacks as CB
import prode.postprocessing as post

from tqdm import tqdm
from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils


@keras_export('keras.callbacks.Plotter')
class Plotter(CB.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        output_ops=None,
        var_names=None,
        time_idx=-1,
        fig_name=None,
        labels=None,
        scales=None,
        limit_y=False,
        log_plot=False,
        freq=100
    ):
        super(Plotter, self).__init__()

        self.output_ops = output_ops

        self.var_names = var_names
        self.labels = labels
        self.time_idx = time_idx

        self.scales = scales
        self.log_plot = log_plot
        self.limit_y = limit_y
        
        self.freq = freq
        self.fig_name = fig_name


    def on_train_begin(self, logs={}):
        self.test_data = self.model.data.test

        self.train_dir = self.model.train_dir
        self.path = {
            'train': self.train_dir+'plots/training',
            'test':  self.train_dir+'plots/testing'
        }
        for k, v in self.path.items():
            if not os.path.exists(v):
                os.makedirs(v)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            utils.print_main(">> `Plotter` callback:")
            self.training(limit_y=self.limit_y)
            if self.test_data is not None:
                self.testing()
            print("")

    def training(self, scale='log', limit_y=False):
        utils.print_submain(" - Plotting training quantities ...")
        history = pd.DataFrame.from_dict(self.model.history.history)
        path = self.path['train']
        # Learning rate
        post.training.plot_lr(self.train_dir+'/lr.csv', path)
        # Losses
        if 'loss' in history:
            post.training.plot_loss(
                history, path, scale=scale, limit_y=limit_y
            )
        elif 'tot_loss' in history:
            post.training.plot_composite_loss(
                history, path, scale=scale, limit_y=limit_y
            )
            if any([ c.endswith('weight') for c in history ]):
                post.training.plot_loss_weights(history, path)
        # Metrics
        if self.model.compiled_metrics:
            for metric in self.model.compiled_metrics.metrics:
                if metric.name in history:
                    post.training.plot_metric(
                        history,
                        path,
                        metric.name,
                        scale=scale,
                        limit_y=limit_y
                    )

    def testing(self):
        utils.print_submain(" - Plotting test cases ...")
        for i, test_i in enumerate(tqdm(copy.deepcopy(self.test_data))):
            # x, y_true = self.unpack_test_data(test_i)
            x, y_true = test_i
            y_pred = self.model.predict(x)
            if self.output_ops:
                y_true, y_pred = list(map(self.output_ops, [y_true, y_pred]))

            path = self.path['test'] + '/test_'+str(i+1)
            if not os.path.exists(path):
                os.makedirs(path)
            if isinstance(y_pred, (list,tuple)):
                if self.fig_name == None:
                    self.fig_name = [
                        'fig'+str(i) for i in range(1,len(y_true)+1)
                    ]
                for j in range(len(y_pred)):
                    kwargs = dict(
                        y_true=y_true[j],
                        y_pred=y_pred[j],
                        var_names=self.var_names[j],
                        time_idx=self.time_idx,
                        fig_name=self.fig_name[j],
                        labels=self.labels[j],
                        scales=self.scales[j]
                    )
                    self.testing_plot_fn(path, x, kwargs)
            else:
                kwargs = dict(
                    y_true=y_true,
                    y_pred=y_pred,
                    var_names=self.var_names,
                    time_idx=self.time_idx,
                    fig_name=self.fig_name,
                    labels=self.labels,
                    scales=self.scales
                )
                self.testing_plot_fn(path, x, kwargs)

    def testing_plot_fn(self, path, x, kwargs):
        post.testing.plot_test(path, x, **kwargs)
        if self.log_plot:
            kwargs['scales'] = ['log','log']
            post.testing.plot_test(path, x,**kwargs)


# @keras_export('keras.callbacks.MovieDumper')
# class MovieDumper(CB.Callback):
#     """Dump a movie to show the training progress of the function along a line.

#     Args:
#         spectrum: If True, dump the spectrum of the Fourier transform.
#     """

#     def __init__(
#         self,
#         test_data=None,
#         train_dir=None,
#         output_ops=None,
#         var_names=None,
#         time_idx=-1,
#         fig_name=None,
#         labels=None,
#         scales=None,
#         limit_y=False,
#         log_plot=False,
#         freq=100


#         self,
#         filename,
#         x1,
#         x2,
#         num_points=100,
#         freq=1,
#         component=0,
#         save_spectrum=False,
#         y_reference=None,
#     ):
#         super(MovieDumper, self).__init__()

#         self.test_data = test_data

#         self.train_dir = train_dir
#         self.path = self.train_dir+'plots/movie'
#         if not os.path.exists(self.path):
#             os.makedirs(self.path)

#         self.output_ops = output_ops

#         self.var_names = var_names
#         self.labels = labels
#         self.time_idx = time_idx

#         self.scales = scales
#         self.log_plot = log_plot
        
#         self.freq = freq
#         self.fig_name = fig_name






#         super(MovieDumper, self).__init__()
#         self.filename = filename
#         x1 = np.array(x1)
#         x2 = np.array(x2)
#         self.x = x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]
#         self.freq = freq
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
#         if self.epochs_since_last_save >= self.freq:
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






# def apply(func, args=None, kwds=None):
#     """Clear Tensorflow GPU memory after model execution.

#     References:

#         - https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
#     """
#     with Pool(1) as p:
#         if args is None and kwds is None:
#             r = p.apply(func)
#         elif kwds is None:
#             r = p.apply(func, args=args)
#         elif args is None:
#             r = p.apply(func, kwds=kwds)
#         else:
#             r = p.apply(func, args=args, kwds=kwds)
#     return r



# def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
#     apply(
#         _save_animation,
#         args=(filename, xdata, ydata),
#         kwds={"y_reference": y_reference, "logy": logy},
#     )


# def save_animation(filename, xdata, ydata, y_reference=None, logy=False):
#     """The animation figure window cannot be closed automatically.

#     References:

#         - https://stackoverflow.com/questions/43776528/python-animation-figure-window-cannot-be-closed-automatically
#     """
#     fig, ax = plt.subplots()
#     if y_reference is not None:
#         plt.plot(xdata, y_reference, "k-")
#     (ln,) = plt.plot([], [], "r-o")

#     def init():
#         ax.set_xlim(np.min(xdata), np.max(xdata))
#         if logy:
#             ax.set_yscale("log")
#             ax.set_ylim(max(1e-4, np.min(ydata)), np.max(ydata))
#         else:
#             ax.set_ylim(np.min(ydata), np.max(ydata))
#         return (ln,)

#     def update(frame):
#         ln.set_data(xdata, ydata[frame])
#         return (ln,)

#     ani = animation.FuncAnimation(
#         fig, update, frames=len(ydata), init_func=init, blit=True
#     )
#     ani.save(filename, writer="imagemagick", fps=30)
#     plt.close()