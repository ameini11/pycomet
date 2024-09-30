import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl

from pathlib import Path
from matplotlib import pyplot as plt

# Use custom style
# See: https://matplotlib.org/1.5.3/users/style_sheets.html
path_to_style = "/home/zanardi/Workspace/Research/styles/matplotlib/"
style_name = "paper_1column"
mpl.style.use(f"{path_to_style}/{style_name}.mplstyle")
# Get colors list
COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]*10


# Stats
# =====================================
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

def error_l2(y_true, y_pred, axis=0):
  err = np.linalg.norm(y_true-y_pred, ord=2, axis=axis)
  return 100.0 * err / np.linalg.norm(y_true, ord=2, axis=axis)

def error_mape(y_true, y_pred, eps=1e-10):
  return 100.0 * np.abs(y_true-y_pred) / (np.abs(y_true)+eps)

# Reading
# =====================================
def load_test_cases(path):
  test = {}
  for file in os.listdir(path):
    filename = os.path.join(path, file)
    test[Path(file).stem] = pd.read_csv(filename, sep=",", header=0)
  return test

# Predict
# =====================================
def predict(model, data, indices):
  x = [np.take(data, i, axis=-1) for i in indices["x"]]
  y_true = np.hstack([np.take(data, i, axis=-1) for i in indices["y"]])
  y_pred = np.hstack(tf.nest.map_structure(lambda x: x.numpy(), model(x)))
  return x[-1], y_true, y_pred

# Plotting
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
  scales=["log", "linear"],
  xlim=None,
  ylim=None,
  save=True,
  show=False
):
  # Initialization
  # -----------------------------------
  fig, ax = plt.subplots()
  # Set up figure
  # -----------------------------------
  # x-axis
  if (labels[0] is not None):
    ax.set_xlabel(labels[0])
  if (scales[0] is not None):
    ax.set_xscale(scales[0])
  if (xlim is not None):
    ax.set_xlim(*xlim)
  elif (scales[0] != "log"):
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
    lw=2.0
  )
  pred_style = dict(
    marker="^",
    fillstyle="none",
    markersize=5,
    markevery=y.shape[0]//20,
    ls="--",
    lw=2.0
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
    idx = np.arange(n)
    ax.legend(
      [plt.plot([], [], c=colors[i])[0] for i in idx],
      [varnames[i] for i in idx],
      # fontsize=20
    )
  # Tight layout
  plt.tight_layout()
  # Finalizing
  # -----------------------------------
  if save:
    for ext in tf.nest.flatten(figext):
      fig.savefig(path + f"/{figname}.{ext}", dpi=300)
  if show:
    plt.show()
  plt.close()
