import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cb

from tensorflow.python.util.tf_export import keras_export
from pycomet import postprocessing
from pycomet import utils
from tqdm import tqdm


@keras_export('keras.callbacks.Plotter')
class Plotter(cb.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        test_data=None,
        train_dir=None,
        output_ops=None,
        var_names=None,
        time_idx=-1,
        fig_name=None,
        labels=None,
        scales=None,
        limit_y=False,
        log_scale=False,
        freq=100
    ):
        super(Plotter, self).__init__()

        self.test_data = test_data

        self.train_dir = train_dir
        self.dir = {
            'train': self.train_dir + 'plots/training',
            'test': self.train_dir + 'plots/testing'
        }
        for k, v in self.dir.items():
            if not os.path.exists(v):
                os.makedirs(v)

        self.output_ops = output_ops

        self.var_names = var_names
        self.labels = labels
        self.time_idx = time_idx

        self.scales = scales
        self.log_scale = log_scale
        self.limit_y = limit_y
        
        self.freq = freq
        self.fig_name = fig_name

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and epoch % self.freq == 0:
            utils._print(">> `Plotter` callback:", lev=0)
            self.training()
            if self.test_data is not None:
                self.testing()
            print("")

    def training(self, scale='log'):
        utils._print(" - Plotting training quantities ...", lev=1)
        history = pd.DataFrame.from_dict(self.model.history.history)
        path = self.dir['train']
        # Learning rate
        postprocessing.training.plot_lr(self.train_dir+'/lr.csv', path)
        # Losses
        if self.model.multi_obj_loss:
            postprocessing.training.plot_composite_loss(
                history, path, scale=scale
            )
            if any([ c.endswith('weight') for c in history.columns ]):
                postprocessing.training.plot_loss_weights(
                    history, path
                )
        else:
            postprocessing.training.plot_loss(history, path, scale=scale)
        # Metrics
        if self.model.compiled_metrics:
            for metric in self.model.compiled_metrics.metrics:
                if metric.name in history:
                    postprocessing.training.plot_metric(
                        history,
                        path,
                        metric.name,
                        scale=scale
                    )

    def testing(self):
        utils._print(" - Plotting test cases ...", lev=1)
        for name, (x, y) in tqdm(copy.deepcopy(self.test_data).items()):
            # Get pred values
            y_pred = self.model.predict(x)
            # Define `testing` directory
            path = self.dir['test'] + f'/{name}'
            if not os.path.exists(path):
                os.makedirs(path)
            # Manipulate model inputs
            if isinstance(x, (list,tuple)):
                x = x[self.time_idx] # Time input
            else:
                x = x[:,self.time_idx]
            # Manipulate model outputs
            if self.output_ops:
                y, y_pred = list(map(self.output_ops, [y, y_pred]))
            # Plot true vs. pred values for each single model output
            if isinstance(y, (list,tuple)):
                for j in range(len(y)):
                    kwargs = dict(
                        path=path,
                        x=x,
                        y=y[j],
                        y_pred=y_pred[j],
                        var_names=self.var_names[j],
                        fig_name=self.fig_name[j] if self.fig_name else None,
                        labels=self.labels,
                        scales=self.scales
                    )
                    self.testing_plot_fn(kwargs)
            else:
                kwargs = dict(
                    path=path,
                    x=x,
                    y=y,
                    y_pred=y_pred,
                    var_names=self.var_names,
                    fig_name=self.fig_name,
                    labels=self.labels,
                    scales=self.scales
                )
                self.testing_plot_fn(kwargs)

    def testing_plot_fn(self, kwargs):
        postprocessing.testing.plot_test(**kwargs)
        if self.log_scale:
            kwargs['scales'] = ['log','log']
            postprocessing.testing.plot_test(**kwargs)
