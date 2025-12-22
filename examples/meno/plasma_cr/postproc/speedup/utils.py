import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import dataframe_image as dfi

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Plotting
# =====================================
# Custom style
# -----------------
# See: https://matplotlib.org/1.5.3/users/style_sheets.html
style_path = "/home/zanardi/Workspace/Research/styles/matplotlib/"
style_name = "paper_1column"
mpl.style.use(f"{style_path}/{style_name}.mplstyle")

# Colors list
# -----------------
def lighten_palette(palette, light_factor=1.15):
  out = []
  for (r, g, b) in palette:
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l * light_factor)
    c = colorsys.hls_to_rgb(h, l, s)
    c = mpl.colors.to_hex(c)
    out.append(c)
  return out

colors = lighten_palette(
  palette=sns.color_palette("Paired")[1::2],
  light_factor=1.1
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
COLORS = colors * 10

# Formatter
# -----------------
class FixedScalarFormatter(ScalarFormatter):

  def __init__(
    self,
    sigfigs=None,
    useMathText=True,
    **kwargs
  ):
    super().__init__(useMathText=useMathText, **kwargs)
    self.sigfigs = sigfigs

  def _set_format(self):
    if (self.sigfigs is None):
      super()._set_format()
    else:
      self.format = f'%.{self.sigfigs}f'
      if (self._usetex or self._useMathText):
          self.format = r'$\mathdefault{%s}$' % self.format

# Plotting functions
# -----------------
def plot_timings_1d(
  x,
  mean,
  std=None,
  var_names=None,
  title=None,
  legend_loc="best",
  labels=[None, None],
  scales=["log", "linear"],
  use_sci=True,
  ysigfigs=2,
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
    if (var_names is not None):
      kwargs["label"] = var_names[i]
    if (std is not None):
      ax.errorbar(x=x, y=mean[i], yerr=std[i], **kwargs)
      ax.fill_between(x=x, y1=mean[i]-std[i], y2=mean[i]+std[i], alpha=0.2)
    else:
      ax.plot(x, mean[i], **kwargs)
  # plt.hlines(1.0, xlim[0], xlim[-1], colors="grey", lw=0.5)
  if use_sci:
    ax.yaxis.set_major_formatter(FixedScalarFormatter(sigfigs=ysigfigs))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0,0))
  if (var_names is not None):
    ax.legend(
      loc=legend_loc,
      fancybox=True,
      framealpha=0.9
    )
  # Save figure
  if save:
    for ext in tf.nest.flatten(extensions):
      fig.savefig(figname+"."+ext)
  # Show and close figure
  if show:
    plt.show()
  plt.close()

def plot_timings_2d(
  xlbl,
  ylbl,
  z,
  figsize=(8,8),
  figname=None,
  save=True,
  show=False,
  extensions=("pdf", "pgf", "png")
):
  fig, ax = plt.subplots(figsize=figsize)
  # Set x-axis
  ax.set_xticks(np.arange(len(xlbl)), labels=xlbl)
  ax.set_xlabel("$t_f$ [s]")
  # Set y-axis
  ax.set_yticks(np.arange(len(ylbl)), labels=ylbl)
  ax.set_ylabel("BDF RTol")
  # Set z-axis
  im = ax.imshow(z, rasterized=True)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.3)
  cbar = fig.colorbar(im, cax=cax, orientation="vertical", label="Speedup")
  cbar.set_ticks(np.linspace(z.min(), z.max(), 5))
  cbar.formatter.set_powerlimits((0, 0))
  # Save figure
  if save:
    for ext in tf.nest.flatten(extensions):
      fig.savefig(figname+"."+ext, bbox_inches="tight", pad_inches=0.1)
  # Show and close figure
  if show:
    plt.show()
  plt.close()


# Speedup computation
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

def compute_speedup(num, den, header):
  # Extract mean and std
  values = {}
  for k in ("mean", "std"):
    values[k] = np.vstack([df[k].to_numpy().squeeze() for df in (num, den)])
  # Compute speedup
  rmean, rstd = ratio_normal_approx(values["mean"], values["std"])
  # Compose data
  time = num["t"].to_numpy().squeeze()
  data = np.vstack([time, rmean, rstd]).T
  return pd.DataFrame(data=data, columns=header)

def as_si(x, ndp=0):
  s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
  m, e = s.split("e")
  fmt = r"{m}\times 10^{{{e:d}}}" if (m != '1') else r"10^{{{e:d}}}"
  return fmt.format(m=m, e=int(e))

def sformat(mean, std):
  return "{:.3f} ".format(mean) + u"\u00B1" + " {:.3f}".format(std)

def save_speedup(df, filename, title=None, savefig=False):
  # Save original data
  df.to_csv(
    path_or_buf=filename + ".csv",
    float_format="%.8e",
    index=False
  )
  # Print original data in a table
  if savefig:
    svalues = []
    sdata = df.to_dict(orient="list")
    for i in range(len(sdata["t"])):
      svalues.append(sformat(sdata["mean"][i], sdata["std"][i]))
    sdata = {
      "t [s]": ["{:.0e}".format(ti) for ti in sdata["t"]],
      "Speedup": svalues
    }
    dfs = pd.DataFrame.from_dict(sdata)
    dfs_styled = dfs.style.set_caption(title).hide(axis="index").background_gradient()
    dfi.export(dfs_styled, filename + "_table.png")
