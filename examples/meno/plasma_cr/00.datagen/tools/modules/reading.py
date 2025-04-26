import os
import numpy as np
import pandas as pd
import dill as pickle
import tensorflow as tf

from typing import *
from tqdm import tqdm
from pycomet.networks.chemkin.plasma_cr.meno import const
from pycomet.networks.chemkin.plasma_cr.meno.mixture import Mixture

# Global variables
SPECIES = ["em"] \
        + [f"Ar({i+1})" for i in range(31)] \
        + [f"Arp({i+1})" for i in range(2)]
HEADER = ["x"] + [f"X_{s}" for s in SPECIES] \
       + ["Th", "Te", "rho", "p", "H", "Mf", "u", "shock_sensor"]


def load_dataset(
  filename: str
) -> Dict[str, np.ndarray]:
  """Load solution from saved file"""
  if os.path.exists(filename):
    with open(filename, "rb") as file:
      return pickle.load(file)
  else:
    raise ValueError(f"File '{filename}' does not exist.")

def read_1d_sim(
  path_to_sol: str,
  path_to_dtb: str,
  savefile: str,
  flow_model: str = "EU_NLTE",
  extension: str = None,
  iter_max: Optional[int] = None,
  save: bool = True,
  species_order: tuple = ("em", "Ar", "Arp")
) -> Dict[str, np.ndarray]:
  """Read solution from HEGEL 1D simulation."""
  # Identify files containing the solution
  # -------------
  filenames, indices = [], []
  flowfield = f"flowfield_{flow_model}"
  if (extension is not None):
    flowfield += "_"+extension
  for filename in sorted(os.listdir(path_to_sol)):
    if filename.startswith(flowfield):
      # Iteration
      index = int(filename.split(".")[0].split("_")[-1])
      indices.append(index)
      # Filename
      filename = os.path.join(path_to_sol, filename)
      filenames.append(filename)
  indices = np.asarray(indices)
  filenames = np.asarray(filenames)
  if (iter_max is not None):
    i = (indices <= iter_max)
    indices = indices[i]
    filenames = filenames[i]
  # Read solutions
  # -------------
  sols = {c: [] for c in HEADER}
  for filename in tqdm(filenames, desc="Iterations", ncols=80):
    idata = pd.DataFrame(data=np.loadtxt(filename), columns=HEADER)
    for c in HEADER:
      sols[c].append(idata[c].values.reshape(-1))
  # > Space grid
  x = sols.pop("x")
  sols["x_grid"] = x[0].reshape(-1)
  # > Time grid
  filename = os.path.join(path_to_sol, f"L2_convergence_{flow_model}.dat")
  t = np.atleast_2d(np.loadtxt(filename))[:,1].reshape(-1)
  t = np.insert(t, 0, 0.0)
  i = np.where(indices <= len(t))[0]
  sols["t_grid"] = t[indices[i]]
  # > Stack sols
  for (k, kdata) in sols.items():
    if (k not in ("x_grid", "t_grid")):
      sols[k] = np.vstack(kdata)
  # Postprocess solutions
  # -------------
  mix = Mixture(
    species={k: path_to_dtb + f"/species/{k}.json" for k in species_order},
    species_order=species_order
  )
  sols["x"] = _get_molar_fractions(sols)
  sols["w"] = _get_mass_fractions(mix=mix, x=sols["x"])
  # Save solutions
  # -------------
  if save:
    with open(savefile, "wb") as f:
      pickle.dump(sols, f)
  return sols

def _get_molar_fractions(sols):
  x = [sols[f"X_{s}"] for s in SPECIES]
  return np.stack(x, axis=-1)

def _get_mass_fractions(mix, x):
  s = x.shape
  x = x.reshape(-1, s[-1])
  w = mix.get_w(x).numpy()
  return w.reshape(s)
