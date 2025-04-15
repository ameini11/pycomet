import os
import numpy as np
import pandas as pd
import dill as pickle

from typing import *
from tqdm import tqdm
from pycomet.networks.chemkin.plasma_cr.meno.mixture import Mixture

# Global variables
HEADER = ["x", "X_em"] \
       + [f"X_Ar({i+1})" for i in range(31)] \
       + [f"X_Arp({i+1})" for i in range(2)] \
       + ["Th", "Te", "rho", "p", "H", "Mf", "u", "shock_sensor"]
SPECIES = ["em"] \
        + [f"Ar({i+1})" for i in range(31)] \
        + [f"Arp({i+1})" for i in range(2)]


def read_dataset(
  path_to_sol: str,
  path_to_dtb: str,
  savefile: str,
  flow_model: str = "EU_NLTE",
  save: bool = True
) -> Dict[str, np.ndarray]:
  """Read solution from HEGEL run"""
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
  filename = os.path.join(path_to_sol, f"L2_convergence_{flow_model}.dat")
  t = np.atleast_2d(np.loadtxt(filename))[:,1].reshape(-1)
  t = np.insert(t, 0, 0.0)
  i = np.where(indices <= len(t))[0]
  sols["t"] = t[indices[i]]
  # > Stack sols
  for (k, kdata) in sols.items():
    if (k not in ("x", "t")):
      sols[k] = np.vstack(kdata)
  # Postprocess solutions
  # -------------
  mix = Mixture(
    {k: path_to_dtb + f"/species/{k}.json" for k in ("Ar", "Arp", "em")}
  )
  sols["X"] = _get_molar_fractions(sols)
  sols["W"] = _get_mass_fractions(mix, sols)
  # Save solutions
  # -------------
  if save:
    with open(savefile, "wb") as f:
      pickle.dump(sols, f)
  return sols

def _get_molar_fractions(sols):
  x = [sols[f"X_{s}"] for s in SPECIES]
  return np.stack(x, axis=-1)

def _get_mass_fractions(mix, sols):
  s = sols["X"].shape
  x = sols["X"].reshape(-1, s[-1])
  w = mix.get_w(x).numpy()
  return w.reshape(s)

def load_dataset(
  filename: str
) -> Dict[str, np.ndarray]:
  """Load solution from saved file"""
  if os.path.exists(filename):
    with open(filename, "rb") as file:
      return pickle.load(file)
  else:
    raise ValueError(f"File '{filename}' does not exist.")
