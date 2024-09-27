import json
import numpy as np
import scipy as sp

from scipy.constants import eV
from scipy.constants import Avogadro as NA


class PlasmaReactor(object):

  def __init__(
    self,
    reactions_file
  ) -> None:
    self.species_name = ("Ar", "Ar4s", "Ar4p", "Ar+", "e-")
    self.species = {name: i for (i, name) in enumerate(self.species_name)}
    self.nb_eqs = len(self.species)
    with open(reactions_file) as file:
      self.reactions = json.load(file)
    self.e_mean = float(self.reactions["mean_ee"]) * eV
    self.rates = self.read_rates()
    self.set_rates_mat()

  # Reactions
  # -----------------------------------
  def read_rates(self):
    kf, kb, indices = [], [], []
    for reaction in self.reactions["reactions"].values():
      kf.append(reaction["kf"])
      kb.append(reaction["kb"])
      ir = []
      for k in ("reac", "prod"):
        for species in reaction[k].keys():
          if (species != "e-"):
            ir.append(self.species[species])
      indices.append(ir)
    return {
      "fwd": np.array(kf),
      "bwd": np.array(kb),
      "indices": np.array(indices).T
    }

  def set_rates_mat(self):
    i = self.rates["indices"]
    K1d = np.zeros(self.nb_eqs**2)
    Ki = np.arange(self.nb_eqs**2).reshape(self.nb_eqs,self.nb_eqs)
    for (rows, cols, sign, direction) in (
      (0, 0, -1, "fwd"),
      (0, 1,  1, "bwd"),
      (1, 0,  1, "fwd"),
      (1, 1, -1, "bwd")
    ):
      indices = Ki[i[rows],i[cols]]
      updates = sign * self.rates[direction]
      np.add.at(K1d, indices, updates)
    self.K = K1d.reshape(self.nb_eqs,self.nb_eqs)
    self.K[-1] = self.K[-2]

  # ODE Solver
  # -----------------------------------
  def solve(self, t, n0, eps=1e-15):
    c = sp.integrate.solve_ivp(
      fun=self.rhs,
      t_span=[0.0,t[-1]],
      y0=n0/NA,
      method="LSODA",
      t_eval=t,
      first_step=1e-16,
      rtol=1e-7,
      atol=eps*1e-1
    ).y.T
    c = np.abs(c) + eps
    return c * NA

  def rhs(self, t, c):
    n = c * NA
    return self.K @ n * n[-1] / NA
