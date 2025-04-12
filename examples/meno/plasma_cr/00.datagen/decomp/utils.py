import os
import sys
import numpy as np
import pandas as pd
import dill as pickle
import tensorflow as tf

from typing import *
from tqdm import tqdm
# from romar.systems import BoxAd
# from romar import postproc as pp
from scipy.constants import physical_constants as pc


from pycomet.networks.chemkin.plasma_cr.meno.mixture import Mixture

import matplotlib
import matplotlib.pyplot as plt
COLORS = 5*matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

# Global variables
HEADER = ["x", "X_em"] \
       + [f"X_Ar({i+1})" for i in range(31)] \
       + [f"X_Arp({i+1})" for i in range(2)] \
       + ["Th", "Te", "rho", "p", "H", "Mf", "u", "shock_sensor"]
SPECIES = ["em"] \
        + [f"Ar({i+1})" for i in range(31)] \
        + [f"Arp({i+1})" for i in range(2)]


# Reading
# =====================================
def read_sol(
  dt: float,
  path_to_sol: str,
  path_to_dtb: str,
  savefile: str,
  flow_model: str = "EU_NLTE",
  load_from_file: bool = False,
  save: bool = True
) -> Dict[str, np.ndarray]:
  if load_from_file:
    # Load solutions from saved file
    # -------------
    if os.path.exists(savefile):
      with open(savefile, "rb") as file:
        sols = pickle.load(file)
    else:
      raise ValueError(f"File '{savefile}' does not exist.")
  else:
    # Identify files containing the solution
    # -------------
    filenames, indices = [], []
    flowfield = f"flowfield_{flow_model}"
    for filename in sorted(os.listdir(path_to_sol)):
      if filename.startswith(flowfield):
        # Iteration
        index = int(filename.split(".")[0].split("_")[-1])
        indices.append(index)
        # Filename
        filename = os.path.join(path_to_sol, filename)
        filenames.append(filename)
    indices = np.asarray(indices)
    # Read solutions
    # -------------
    sols = {c: [] for c in HEADER}
    for filename in tqdm(filenames, desc="Iterations", ncols=80):
      idata = pd.DataFrame(data=np.loadtxt(filename), columns=HEADER)
      for c in HEADER:
        sols[c].append(idata[c].values.reshape(-1))
    # > Space grid
    sols["x"] = sols["x"][0].reshape(-1)
    # > Time grid
    sols["t"] = dt * indices
    # > Stack sols
    for (k, kdata) in sols.items():
      if (k not in ("x", "t")):
        sols[k] = np.vstack(kdata)
    # Postprocess solutions
    # -------------
    mix = Mixture(
      {k: path_to_dtb + f"/species/{k}.json" for k in ("Ar", "Arp", "em")}
    )
    sols["x"], sols["w"] = _get_mass_fractions(mix, sols)
    # Save solutions
    # -------------
    if save:
      with open(savefile, "wb") as f:
        pickle.dump(sols, f)
  return sols

def _get_mass_fractions(mix, sols):
  x = [sols[f"X_{s}"] for s in SPECIES]
  x = np.stack(x, axis=-1)
  s = x.shape
  x = x.reshape(-1, s[-1])
  w = mix.get_w(x).numpy()
  return x.reshape(s), w.reshape(s)

# # Induction length
# # =====================================
# def get_index_shock_loc(
#   p: np.ndarray
# ) -> np.ndarray:
#   """
#   Identify the index of the shock front location for each row of a
#   2D time-space pressure array.

#   :param p: 2D array of pressure values.
#   :return: 1D array of indices marking shock front location.
#   """
#   mask = np.zeros_like(p)
#   mask[p > p[0,0]] = 1.0
#   i = np.asarray([np.argmax(m) for m in mask])
#   i[i == 0] = -1
#   return i

# def get_index_peak_el(
#   x_em: np.ndarray
# ) -> np.ndarray:
#   """
#   Returns the index of the first peak in electron molar fraction for each
#   row (i.e., time instant). The peak is identified as the first maximum
#   followed by a negative gradient.

#   :param x_em: 2D time-space array of electron molar fractions.
#   :return: 1D array of peak indices.
#   """
#   dx = np.gradient(x_em, axis=-1, edge_order=2)
#   i = np.zeros(len(x_em), dtype=int)
#   for j, dxj in enumerate(dx):
#     l1 = np.argmax(dxj)
#     l2 = np.where(dxj[l1:] < 0.0)[0]
#     if (len(l2) == 0):
#       i[j] = -1
#     else:
#       i[j] = l2[0] + l1
#   return i

# def compute_induct(
#   x: np.ndarray,
#   sol: dict
# ) -> np.ndarray:
#   """
#   Compute the induction length (distance between shock front and electron peak).

#   :param x: Spatial grid (1D).
#   :param sol: Dictionary containing 'p' and 'X_em' 2D arrays.
#   :return: 1D array of induction lengths in centimeters.
#   """
#   i = get_index_shock_loc(p=sol["p"])
#   j = get_index_peak_el(x_em=sol["X_em"])
#   return x[i] - x[j]

# def get_indices_induct_peaks(
#   t: np.ndarray,
#   l: np.ndarray,
#   delta_min: int = 15
# ) -> np.ndarray:
#   """
#   Identify local maxima in the induction length signal.

#   :param t: Time array.
#   :param l: Induction length array.
#   :param delta_min: Minimum index spacing between peaks.
#   :return: Indices of peaks satisfying spacing constraint.
#   """
#   dl = np.sign(np.gradient(l, t, edge_order=2))
#   indices = []
#   sm = dl[0]
#   for (i, sp) in enumerate(dl[1:]):
#     if ((sm == 1) and (sp == -1)):
#       indices.append(i)
#     sm = sp
#   indices = np.asarray(indices) + 1
#   return indices[1:][np.diff(indices) > delta_min]

# def compute_average_induct(
#   t: np.ndarray,
#   l: np.ndarray,
#   delta_min: int = 15
# ) -> np.ndarray:
#   """
#   Compute average induction time and average induction length.

#   :param t: Time array.
#   :param l: Induction length array.
#   :param delta_min: Minimum index spacing between peaks.
#   :return: Indices of peaks satisfying spacing constraint.
#   """
#   indices = get_indices_induct_peaks(t, l, delta_min)
#   tmean = np.mean(np.diff(t[indices]))
#   lmean = []
#   for i, j in enumerate(indices[:-1]):
#     k = indices[i+1]+1
#     li = np.trapz(y=l[j:k], x=t[j:k])
#     li /= (t[k] - t[j])
#     lmean.append(li)
#   lmean = np.mean(lmean)
#   return tmean, lmean

# # Plotting
# # =====================================
# def plot_t_sol(
#   t,
#   y,
#   labels,
#   yexp=None,
#   ylim=None,
#   ylabel=r"$n$ [m$^{-3}$]",
#   yscale="linear",
#   legend_loc="best",
#   use_sci=False,
#   sigfigs=2,
#   figname=None,
#   save=False,
#   show=False
# ):
#   # Initialize figures
#   fig = plt.figure()
#   ax = fig.add_subplot()
#   # x axis
#   t = 1e3*t
#   ax.set_xlabel(r"$t$ [ms]")
#   ax.set_xscale("linear")
#   tmin, tmax = np.amin(t), np.amax(t)
#   ax.set_xlim((tmin, tmax))
#   # y axis
#   ax.set_ylabel(ylabel)
#   ax.set_yscale(yscale)
#   if (ylim is not None):
#     ax.set_ylim(ylim)
#   # Plotting
#   for (k, yk) in y.items():
#     if (k.upper() == "FOM"):
#       style = dict(ls="-", lw=1.5)
#     else:
#       style = dict(ls="--", lw=3.0)
#     ax.plot(t, yk, c="k", label=labels[k], **style, rasterized=True)
#   if (yexp is not None):
#     ax.hlines(yexp, tmin, tmax, ls="--", colors="k", lw=1.0, label="Experiments")
#   ax.legend(loc=legend_loc, fancybox=True, framealpha=0.9)
#   if use_sci:
#     ax.yaxis.set_major_formatter(pp.FixedScalarFormatter(sigfigs=sigfigs))
#     ax.yaxis.get_major_formatter().set_scientific(True)
#     ax.yaxis.get_major_formatter().set_powerlimits((0,0))
#   # Tight layout
#   plt.tight_layout()
#   if save:
#     plt.savefig(figname)
#   if show:
#     plt.show()
#   plt.close()

# def plot_x_sol(
#   x,
#   y,
#   xlim,
#   labels,
#   ylabel=r"$n$ [m$^{-3}$]",
#   yscale="linear",
#   legend_loc="best",
#   use_sci=False,
#   sigfigs=2,
#   figname=None,
#   save=False,
#   show=False
# ):
#   # Initialize figures
#   fig = plt.figure()
#   ax = fig.add_subplot()
#   # x axis
#   ax.set_xlabel(r"$x$ [cm]")
#   ax.set_xscale("linear")
#   ax.set_xlim(xlim)
#   ax.invert_xaxis()
#   # y axis
#   ax.set_ylabel(ylabel)
#   ax.set_yscale(yscale)
#   # Plotting
#   styles = {}
#   _lines, _labels = [], []
#   for (k, yk) in y.items():
#     if (k.upper() == "FOM"):
#       styles[k] = dict(ls="-", marker="", lw=1.5)
#     else:
#       styles[k] = dict(ls="--", marker="", lw=3.0)
#     # Legend
#     _labels.append(labels[k])
#     _lines.append(ax.plot([], [], c="k", markersize=6, **styles[k])[0])
#   ax.legend(_lines, labels=_labels, loc=legend_loc, fancybox=True, framealpha=0.9)
#   for (k, yk) in y.items():
#     nb_lines = len(yk)
#     for (i, yi) in enumerate(yk):
#       c = "k" if (nb_lines == 1) else COLORS[i]
#       ax.plot(100*x, yi, c=c, markersize=6, markevery=50, **styles[k], rasterized=True)
#   if use_sci:
#     ax.yaxis.set_major_formatter(pp.FixedScalarFormatter(sigfigs=sigfigs))
#     ax.yaxis.get_major_formatter().set_scientific(True)
#     ax.yaxis.get_major_formatter().set_powerlimits((0,0))
#   # Tight layout
#   plt.tight_layout()
#   if save:
#     plt.savefig(figname)
#   if show:
#     plt.show()
#   plt.close()

# def plot_xt_sol(
#   X: np.ndarray,
#   Y: np.ndarray,
#   Z: np.ndarray,
#   xlim: list,
#   ylim: list,
#   zlim: list,
#   label: str,
#   figname: str,
#   sigfigs: Optional[int] = None,
#   shading: str = "gouraud" # "nearest" "gouraud"
# ) -> None:
#   plt.figure()
#   mesh = plt.pcolormesh(
#     X, Y, Z,
#     shading=shading,
#     cmap="jet",
#     edgecolors="none",
#     vmin=zlim[0], vmax=zlim[1],
#     rasterized=True
#   )
#   # plt.contourf(X, Y, Z, 50, cmap="jet", vmin=zlim[0], vmax=zlim[1])
#   cbar = plt.colorbar(mesh, label=label, pad=0.03)
#   cbar.formatter = pp.FixedScalarFormatter(sigfigs=sigfigs)
#   cbar.formatter.set_scientific(True)
#   cbar.formatter.set_powerlimits((0,0))
#   cbar.update_ticks()
#   plt.xlim(xlim)
#   plt.ylim(ylim)
#   plt.xlabel(r"$x$ [cm]")
#   plt.ylabel(r"$t$ [ms]")
#   plt.gca().invert_xaxis()
#   plt.axis('off')
#   plt.savefig(figname, dpi=300)
#   plt.close()
