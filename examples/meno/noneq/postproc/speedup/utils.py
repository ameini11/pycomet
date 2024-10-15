import numpy as np
import tensorflow as tf
import matplotlib as mpl

from matplotlib import pyplot as plt

# Use custom style
# See: https://matplotlib.org/1.5.3/users/style_sheets.html
path_to_style = "/home/zanardi/Workspace/Research/styles/matplotlib/"
style_name = "paper_1column"
mpl.style.use(f"{path_to_style}/{style_name}.mplstyle")
# Get colors list
COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]*10


def ratio_normal_approx(mean, std):
  """
  Normal approximation to the distribution of the ratio of two independent 
  normal random variables.
  > See:
    https://link.springer.com/article/10.1007/s00362-012-0429-2 (Eq. 7-8)
  """
  rmean, deltas = mean[0]/mean[1], std/mean
  rstd = np.sqrt(rmean**2 * (deltas[0]**2 + deltas[1]**2))
  return rmean, rstd

def plot_timings(
  x,
  mean,
  std=None,
  var_names=None,
  title=None,
  legend_loc="best",
  labels=[None, None],
  scales=["log", "linear"],
  figname=None,
  save=True,
  show=False,
  extensions=("pdf", "pgf", "png")
):
  fig, ax = plt.subplots()
  # Set up figure
  if (title is not None):
    ax.set_title(title)
  # Set axis
  x_label, y_label = labels
  x_scale, y_scale = scales
  # X axis
  if (x_label is not None):
    ax.set_xlabel(x_label)
  if (x_scale is not None):
    ax.set_xscale(x_scale)
  xlim = [x[0]*0.9, x[-1]*1.1]
  ax.set_xlim(xlim)
  # Y axis
  if (y_label is not None):
    ax.set_ylabel(y_label)
  if (y_scale is not None):
    ax.set_yscale(y_scale)
  # Plotting
  kwargs = dict(linestyle="--", marker="s")
  for i in range(len(mean)):
    if (std is not None):
      ax.errorbar(x=x, y=mean[i], yerr=std[i], label=var_names[i], **kwargs)
      ax.fill_between(x=x, y1=mean[i]-std[i], y2=mean[i]+std[i], alpha=0.2)
    else:
      ax.plot(x, mean[i], label=var_names[i], **kwargs)
  # plt.hlines(1.0, xlim[0], xlim[-1], colors="grey", lw=0.5)
  ax.legend(loc=legend_loc)
  # Save figure
  if save:
    for ext in tf.nest.flatten(extensions):
      fig.savefig(figname+"."+ext)
  # Show and close figure
  if show:
    plt.show()
  plt.close()
