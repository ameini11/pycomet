import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cb

from tensorflow.python.util.tf_export import keras_export
from pycomet import postproc
from pycomet import utils
from tqdm import tqdm


@keras_export('keras.callbacks.Plotter')
class Plotter(cb.Callback):
    """Plot variables in order to check the training progress."""

    def __init__(
        self,
        test_data=None,
        train_dir=None,
        figures=[],
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

        self.figures = tf.nest.flatten(figures)
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and epoch % self.freq == 0:
            utils.print(">> `Plotter` callback:", lev=0)
            self.training()
            if self.test_data is not None:
                self.testing()
            print("")

    def training(self, scale='log'):
        utils.print(" - Plotting training quantities ...", lev=1)
        history = pd.DataFrame.from_dict(self.model.history.history)
        path = self.dir['train']
        # Learning Rate
        postproc.training.plot_lr(
            history=self.train_dir+'/lr.csv', path=path
        )
        for scale in tf.nest.flatten(scales):
            # Losses
            if self.net.multi_obj_loss:
                postproc.training.plot_composite_loss(
                    history, path, scale=scale
                )
            else:
                postproc.training.plot_loss(
                    history, path, scale=scale
                )
            # Metrics
            if self.net.compiled_metrics is not None:
                for metric in self.net.compiled_metrics.metrics:
                    if metric.name in history:
                        postproc.training.plot_metric(
                            history,
                            path,
                            metric.name,
                            scale=scale
                        )
        # Weights
        if self.net.multi_obj_loss and any(
            [c.endswith('weight') for c in history.columns]
        ):
            postproc.training.plot_loss_weights(history, path)

    def testing(self):
        utils.print(" - Plotting test cases ...", lev=1)
        for name, (x, y) in tqdm(copy.deepcopy(self.test_data).items()):
            # Get pred values
            y_pred = self.model.predict(x)
            # Define `testing` directory
            path = self.dir['test'] + f'/{name}'
            if not os.path.exists(path):
                os.makedirs(path)
            # Loop over figures
            for figure in self.figures:
                # Extract data
                _x, _y, _y_pred = tf.nest.map_structure(
                    ops.extract_data, [x, y, y_pred], figure['indeces']
                )
                # Manipulate y-axis values
                if 'output_ops' in figure:
                    _y, _y_pred = list(
                        map(figure['output_ops'], [_y, _y_pred])
                    )
                # Arguments for the plotting function
                plot_kwargs = dict(path=path, x=_x, y=_y, y_pred=_y_pred)
                for k in (
                    'fig_name', 'var_names', 'labels', 'scales', 'title', \
                    'xlim', 'ylim', 'save', 'fig_ax', 'max_num_var'
                ):
                    if k in figure:
                        plot_kwargs.update({k: figure[k]})
                # Plotting
                postproc.testing.plot_test(**plot_kwargs)
