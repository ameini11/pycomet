import numpy as np
import pandas as pd
import dill as pickle
import tensorflow as tf

from typing import Dict
from pycomet.networks.elements import LinAutoencoder
from pycomet.networks.chemkin.plasma_cr.meno import const, Mixture


class Converter(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    path_to_dtb,
    rom_config=None,
    species_order=("em", "Ar", "Arp")
  ):
    # Mixture model
    self.mix = Mixture(
      species={k: path_to_dtb + f"/species/{k}.json" for k in species_order},
      species_order=species_order
    )
    # Linear autoencoder
    self.rom = None
    if (rom_config is not None):
      if isinstance(rom_config, str):
        with open(rom_config, "rb") as file:
          rom_config = pickle.load(file)
      self.rom = LinAutoencoder(**rom_config)
    # Primitive labels
    self.species_names = []
    for k in self.mix.species_order:
      nc = self.mix.species[k].nb_comp
      names = [k] if (nc == 1) else [k+f"{i+1}" for i in range(nc)]
      self.species_names.extend(names)
    self.prim_lbl = [f"rho_{s}" for s in self.species_names] + ["Th", "Te"]

  # Primitive to physical variables
  # -----------------------------------
  def prim_to_phys(
    self,
    t: np.ndarray,
    y: np.ndarray
  ) -> Dict[str, np.ndarray]:
    # Manipulate primitive variables
    nc = self.mix.nb_comp
    rho, Th, Te = np.split(y, [nc, nc+1], axis=-1)
    n = self.mix.get_n(rho).numpy()
    # Store physical variables
    phys = {
      "t": t,
      "Th": Th,
      "Te": Te,
      "n": np.sum(n, axis=-1),
      "rho": np.sum(rho, axis=-1)
    }
    for sp in self.mix.species.values():
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

  # Input space to primitive variables
  # -----------------------------------
  def inp_to_prim(
    self,
    inp: pd.DataFrame
  ) -> pd.DataFrame:
    # Decode latent space
    z = inp[[f"z{i+1}" for i in range(self.rom.dim)]].values
    w = self.rom(z, method=2).numpy()
    # Enforce conservation: mass and charge
    w = self._enforce_cons_mass(w)
    w = self._enforce_cons_charge(w)
    # Collect primitive variables
    rho = w * inp[["rho"]].values
    Th, dT = [inp[[k]].values for k in ("Th", "dT")]
    Te = Th - dT
    prim = np.hstack([rho, Th, Te])
    return pd.DataFrame(prim, columns=self.prim_lbl)

  def _enforce_cons_mass(
    self,
    z: np.ndarray
  ) -> np.ndarray:
    i = self.mix.species["Ar"].indices[0]
    z[:,i] = 0.0
    z[:,i] = 1.0 - np.sum(z, axis=-1)
    return z

  def _enforce_cons_charge(
    self,
    w: np.ndarray
  ) -> np.ndarray:
    x = self.mix.get_x(w).numpy()
    ie, ii = [self.mix.species[k].indices for k in ("em", "Arp")]
    x[:,ie] = np.sum(x[:,ii], axis=-1, keepdims=True)
    x = self._enforce_cons_mass(x)
    w = self.mix.get_w(x).numpy()
    return w
