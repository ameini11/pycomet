import os
import copy
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cb

from tqdm import tqdm
from pycomet import utils
from pycomet import postproc
from pycomet import operations as ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.callbacks.Plotter')
class Plotter(cb.Callback):

  # Initialization
  # ===================================
  def __init__(
    self,
    test_data=None,
    train_dir=None,
    figures=[],
    freq=100,
    max_nb=None
  ):
    super(Plotter, self).__init__()
    self.test_data = test_data
    self.train_dir = train_dir
    self.dir = {
      'train': self.train_dir + 'plots/training',
      'test': self.train_dir + 'plots/testing'
    }
    for path in self.dir.values():
      os.makedirs(path, exist_ok=True)
    self.figures = figures
    self.freq = freq
    self.max_nb = max_nb

  # Callback methods
  # ===================================
  def on_epoch_end(self, epoch, logs=None):
    if ((epoch != 0) and (epoch % self.freq == 0)):
      utils.print_fn(">> `Plotter` callback:", start='\n\n', lev=0)
      self.training()
      if (self.test_data is not None):
        self.testing()
      print("")

  def training(self, scales='log'):
    utils.print_fn("- Plotting training quantities ...", lev=1)
    history = pd.DataFrame.from_dict(self.model.history.history)
    path = self.dir['train']
    # Learning Rate
    postproc.training.plot_lr(
      history=self.train_dir+'/lr.csv', path=path
    )
    for scale in tf.nest.flatten(scales):
      # Losses
      if self.model.multi_obj_loss:
        postproc.training.plot_composite_loss(history, path, scale=scale)
      else:
        postproc.training.plot_loss(history, path, scale=scale)
      # Metrics
      if (self.model.compiled_metrics is not None):
        for metric in self.model.compiled_metrics.metrics:
          if (metric.name in history):
            postproc.training.plot_metric(
              history,
              path,
              metric.name,
              scale=scale
            )
    # Weights
    c1 = self.model.multi_obj_loss
    c2 = any([c.endswith('weight') for c in history.columns])
    if (c1 and c2):
      postproc.training.plot_loss_weights(history, path)

  def testing(self):
    utils.print_fn("- Plotting test cases ...", lev=1)
    data = copy.deepcopy(self.test_data)
    for (i, (name, (x, y))) in enumerate(tqdm(data.items())):
      if ((self.max_nb is not None) and (i+1 > self.max_nb)):
        break
      # Get pred values
      y_pred = self.model.predict(x, verbose=0)
      # Define `testing` directory
      path = self.dir['test'] + f'/{name}'
      os.makedirs(path, exist_ok=True)
      # Loop over figures
      for figure in self.figures:
        # Extract data
        _x, _y, _y_pred = list(map(
          lambda x,i: ops.extract_data(x,i,1),
          [x, y, y_pred],
          figure['indices']
        ))
        # Manipulate y-axis values
        if ('output_ops' in figure):
          _y, _y_pred = list(map(figure['output_ops'], [_y, _y_pred]))
        # Arguments for the plotting function
        plot_kwargs = dict(path=path, x=_x, y=_y, y_pred=_y_pred)
        for k in (
          'fig_name', 'var_names', 'labels', 'scales', 'title', \
          'xlim', 'ylim', 'save', 'fig_ax', 'max_num_var'
        ):
          if (k in figure):
            plot_kwargs.update({k: figure[k]})
        # Plotting
        postproc.testing.plot_test(**plot_kwargs)
