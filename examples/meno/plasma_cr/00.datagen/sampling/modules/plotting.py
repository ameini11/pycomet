import numpy as np
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']*10


# Cumulative energy
def cumenergy(
  y,
  figname=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel('$k$-th basis')
  # y axis
  ax.set_ylabel('Cumulative energy')
  # Plotting
  ax.plot(
    x,
    y,
    markeredgewidth=1,
    lw=1.0,
    markersize=5.0,
    marker='o'
  )
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  else:
    plt.close()


def plot_pairwise(
  X,
  names,
  scales=None,
  xlim=None,
  figsize=(8.0, 8.0),
  labelpad=None,
  figname=None,
  show=False
):
  X = tf.nest.flatten(X)
  n = len(X[0])-1
  fig, axs = plt.subplots(
    n, n,
    sharex="col",
    sharey="row",
    figsize=figsize
  )
  axs = np.array(axs).reshape(n,n)
  for i in range(n):
    for j in range(n):
      if (i < j):
        axs[i,j].axis("off")
      else:
        if (j == 0):
          axs[i,j].set_ylabel(names[i+1], fontsize=12)
        if (i == n-1):
          axs[i,j].set_xlabel(names[j], fontsize=12)
        if (xlim is not None):
          axs[i,j].set_xlim([xlim[0,j], xlim[1,j]])
          axs[i,j].set_ylim([xlim[0,i+1], xlim[1,i+1]])
        if (scales is not None):
          axs[i,j].set_xscale(scales[j])
          axs[i,j].set_yscale(scales[i+1])
        else:
          axs[i,j].ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0,0)
          )
          if (labelpad is not None):
            axs[i,j].xaxis.labelpad = labelpad[0]
            axs[i,j].yaxis.labelpad = labelpad[1]
        for x in X:
          axs[i,j].plot(
            x[j],
            x[i+1],
            marker="o",
            markersize=1,
            ls="",
            fillstyle="full"
          )
  plt.tight_layout()
  fig.savefig(figname)
  if show:
    plt.show()
  plt.close()