import numpy as np

from pycomet import utils
matplotlib, plt = utils.import_matplotlib()
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


# Plotting testing quantities
# =====================================
def plot_test(
  path=None,
  x=None,
  y=None,
  y_pred=None,
  varnames=None,
  figname=None,
  figext="png",
  labels=[None, None],
  scales=['log', 'linear'],
  title=None,
  xlim=None,
  ylim=None,
  save=True,
  fig_ax=None,
  max_num_var=10
):
  # Initialization
  # -----------------------------------
  if (fig_ax is None):
    fig, ax = plt.subplots()
  else:
    fig, ax = fig_ax
  # Set up figure
  # -----------------------------------
  if (title is not None):
    ax.set_title(title)
  # x-axis
  if (labels[0] is not None):
    ax.set_xlabel(labels[0])
  if (scales[0] is not None):
    ax.set_xscale(scales[0])
  if (xlim is not None):
    ax.set_xlim(xlim)
  elif (scales[0] != 'log'):
    ax.set_xlim([min(x), max(x)])
  # y-axis
  if (labels[1] is not None):
    ax.set_ylabel(labels[1])
  if (scales[1] is not None):
    ax.set_yscale(scales[1])
  if (ylim is not None):
    ax.set_ylim(ylim)
  # Check 2nd dim: looping over columns
  x = np.squeeze(x)
  if (x.ndim < 2):
    x = np.expand_dims(x, 1)
  y = np.squeeze(y)
  if (y.ndim < 2):
    y = np.expand_dims(y, 1)
  y_pred = np.squeeze(y_pred)
  if (y_pred.ndim < 2):
    y_pred = np.expand_dims(y_pred, 1)
  # Plot figure
  # -----------------------------------
  # Define styles
  true_style = dict(
    lw=1.
  )
  pred_style = dict(
    marker='^',
    fillstyle='none',
    markersize=5,
    markevery=y.shape[0]//20,
    ls='--',
    lw=1.
  )
  # Define parameters
  n = y.shape[1]
  # Check number of colors available
  colors = COLORS
  if (n > len(COLORS)):
    colors *= int(np.ceil(n/len(COLORS)))
  # Loop over features
  for d in range(n):
    # > Plot true values
    ax.plot(
      x,
      y[:,d],
      c=colors[d],
      **true_style
    )
    # > Plot predicted values
    ax.plot(
      x,
      y_pred[:,d],
      **pred_style,
      c=colors[d]
    )
  # Add legend
  if (varnames is not None):
    idx = np.arange(0,n)
    if (n > 6):
      if (n > max_num_var):
        step = int(np.ceil(n/max_num_var))
        idx = np.arange(0,n,step)
        n = max_num_var
      fs = 'small'
    else:
      fs = 'medium'
    ax.legend(
      [plt.plot([], [], c=colors[i])[0] for i in idx],
      [varnames[i] for i in idx],
      fontsize=fs,
      ncol=int(np.ceil(n/3))
    )
  # Save figure
  # -----------------------------------
  if save:
    figname = f'output_{scales[1]}' if (figname is None) else figname
    fig.savefig(path + f'/{figname}.{figext}')
  # Finalizing
  # -----------------------------------
  if (fig_ax is not None):
    return fig, ax
  plt.close()
