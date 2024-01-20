import numpy as np
import pandas as pd

from pycomet import utils
matplotlib, plt = utils.import_matplotlib()


# Plotting training quantities
# =====================================
def plot_loss(
  history=None,
  path=None,
  scale='log',
  title='Loss',
  variables=['loss','val_loss'],
  figname='loss',
  figext='png',
  label=['Training', 'Validation'],
  window=1
):
  fig = plt.figure()
  plt.yscale(scale)
  for (i, ivar) in enumerate(variables):
    if (ivar in history):
      plt.plot(
        _moving_average(history[ivar], w=window),
        lw=1,
        label=label[i]
      )
      if (window != 1):
        plt.plot(
          history[ivar],
          lw=1,
          c=plt.gca().lines[-1].get_color(),
          alpha=0.2
        )
  plt.xlabel('Epoch')
  plt.legend()
  plt.title(title + ' History')
  fig.savefig(path + f'/{figname}.{figext}')
  plt.close()

def _moving_average(x, w=20):
  return np.convolve(x, np.ones(w), 'valid') / w

def plot_composite_loss(
  history,
  path,
  scale='log',
  window=1
):
  identifiers = {
    'ics_loss': 'ICs Loss',
    'pts_loss': 'Anchor pts Loss',
    'res_loss': 'Residual Loss',
    'reg_loss': 'Regularization Loss',
    'loss': 'Loss'
  }
  kwargs = dict(
    history=history,
    path=path,
    scale=scale,
    window=window
  )
  composite_loss = {'variables': [], 'label': []}
  for c in history.columns:
    if ((c.endswith('loss') and (not c.startswith('val')))):
      if (c not in ('loss', 'weighted_loss')):
        composite_loss['variables'].append(c)
        composite_loss['label'].append(identifiers[c])
      if (c not in ('reg_loss', 'weighted_loss')):
        plot_loss(
          title=identifiers[c],
          variables=[c,'val_'+c],
          figname=c,
          **kwargs
        )
  plot_loss(
    **composite_loss,
    title='Losses',
    figname='losses',
    **kwargs
  )

def plot_loss_weights(
  history,
  path,
  scale='log',
  figext='png'
):
  fig = plt.figure()
  plt.yscale(scale)
  labels = {
    'ics': 'ICs',
    'pts': 'Anchor pts',
    'res': 'Residual',
  }
  for c in history.columns:
    if c.endswith('weight'):
      weight_id = c.split("_")[0]
      plt.plot(history[c], lw=1, label=labels[weight_id])
  plt.xlabel('Epoch')
  plt.legend()
  plt.title('Loss Weights')
  fig.savefig(path + f'/loss_weights.{figext}')
  plt.close()

def plot_metric(
  history,
  path,
  metric,
  scale='log',
  figext='png'
):
  fig = plt.figure()
  plt.yscale(scale)
  plt.plot(history[[metric]], lw=1, label='Training')
  if ('val_'+metric in history):
    plt.plot(history[['val_'+metric]], lw=1, label='Validation')
  plt.xlabel('Epoch')
  plt.title(
    r'$\it{%s}$' % metric.replace("_", "\ ").title() + '  Metric History'
  )
  plt.legend()
  fig.savefig(path + f'/metric_{metric}.{figext}')
  plt.close()

def plot_lr(
  history,
  path,
  figext='png'
):
  df = pd.read_csv(history)
  fig = plt.figure()
  plt.plot(df[['lr']], 'r', lw=1)
  plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
  plt.xlabel('Epoch')
  plt.title('Learning Rate History')
  fig.savefig(path + f'/lr.{figext}')
  plt.close()
