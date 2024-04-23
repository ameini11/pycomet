import tensorflow as tf

import matplotlib as mpl
from matplotlib import pyplot as plt

# Use custom style
style_name = 'paper_1column'
mpl.style.use(
  f"/home/zanardi/Workspace/Research/styles/matplotlib/{style_name}.mplstyle"
)
# Get colors list
COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']*10


# Speedup/Computational time
# =====================================
def timings(
  x,
  mean,
  std,
  var_names=None,
  title=None,
  legend_loc='best',
  labels=[None, None],
  scales=['log', 'linear'],
  figname=None,
  save=True,
  show=False,
  extensions=('pdf', 'pgf', 'png')
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
  kwargs = dict(linestyle='--', marker='s')
  for i in range(len(mean)):
    ax.errorbar(x=x, y=mean[i], yerr=std[i], label=var_names[i], **kwargs)
    ax.fill_between(x=x, y1=mean[i]-std[i], y2=mean[i]+std[i], alpha=0.2)
  # plt.hlines(1.0, xlim[0], xlim[-1], colors='grey', lw=0.5)
  ax.legend(loc=legend_loc)
  # Save figure
  if save:
    for ext in tf.nest.flatten(extensions):
      fig.savefig(figname+'.'+ext)
  # Show and close figure
  if show:
    plt.show()
  plt.close()
