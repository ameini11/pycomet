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
  save: bool = True,
  species_order: tuple = ("em", "Ar", "Arp")
) -> Dict[str, np.ndarray]:
  """Read solution from HEGEL 1D simulation."""
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

def prim_to_phys(mix, t, y):
  # Manipulate primitive variables
  nc = mix.nb_comp
  rho, Th, Te = np.split(y, [nc, nc+1], axis=-1)
  n = mix.get_n(rho).numpy()
  # Store physical variables
  phys = {
    "t": t,
    "Th": Th,
    "Te": Te,
    "n": np.sum(n, axis=-1),
    "rho": np.sum(rho, axis=-1)
  }
  for sp in mix.species.values():
    i = sp.indices.numpy()
    phys[f"n_{sp.name}"] = np.sum(n[:,i], axis=-1)
    phys[f"rho_{sp.name}"] = np.sum(rho[:,i], axis=-1)
    if (sp.name == "Ar"):
      phys["n_Ar1"] = n[:,i[0]]
  phys = tf.nest.map_structure(lambda x: x.reshape(-1,1), phys)
  # Molar/mass fractions
  phys["x"] = n / phys["n"]
  phys["w"] = rho / phys["rho"]
  # Pressures
  ne = phys["n_em"]
  nh = phys["n_Ar"] + phys["n_Arp"]
  phys["pe"] = ne * const.UKB * Te
  phys["ph"] = nh * const.UKB * Th
  phys["p"] = phys["pe"] + phys["ph"]
  # Return
  return phys
