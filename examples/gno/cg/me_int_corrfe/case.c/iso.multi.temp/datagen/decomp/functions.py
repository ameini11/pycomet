import numpy as np
import matplotlib.pyplot as plt


# Plotting
# =====================================
style = dict(
  markeredgewidth=1,
  lw=1.0,
  linestyle='',
  markersize=1.0,
  marker='x'
)

# 2D distribution
def plot_2d(
  x,
  y_true,
  y_pred=None,
  labels=[r'$\epsilon_i$ [eV]', r'$n_i/g_i$ [m$^{-3}$]'],
  scales=['linear', 'log'],
  filename=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  xlim = [np.amin(x), np.amax(x)]
  ax.set_xlim(xlim)
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  ax.plot(x, y_true, c='k', lw=1)
  if (y_pred is not None):
    ax.plot(x, y_pred, c='r', lw=1)
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  else:
    plt.close()

def plot_traj(
  xy=None,
  labels=[r'$t$ [s]', r'$X$'],
  scales=['log', 'linear'],
  figname=None,
  save=False,
  show=True
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  for (xi, yi) in xy:
    ax.plot(xi, yi, lw=1)
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  else:
    plt.close()

# Cumulative energy
def plot_cumenergy(
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

# Fitting
# =====================================
def ols(x, y):
  # Ordinary least squares (OLS) algorithm
  # > Gram matrix
  gr = np.matmul(x.T, x)
  # > Inverse of Gram matrix
  gri = np.linalg.pinv(gr)
  # > Moment matrix
  ym = np.matmul(x.T, y)
  # > Compute coefficients
  w = np.matmul(gri, ym)
  return w
