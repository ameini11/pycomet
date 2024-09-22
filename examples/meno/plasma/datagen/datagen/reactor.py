import json
import numpy as np
import scipy as sp
import pycomet as pcm

from scipy.constants import eV
from scipy.constants import Planck as hp
from scipy.constants import Rydberg as Ry
from scipy.constants import Avogadro as NA
from scipy.constants import electron_mass as me
from scipy.constants import elementary_charge as qe
from scipy.constants import speed_of_light as c_spd
from scipy.constants import physical_constants

rB = physical_constants["Bohr radius"][0]
EI = hp*c_spd*Ry/qe # Ionization energy of Hydrogen (eV)


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

  # Cross sections
  # -----------------------------------
  def cross_sec_ion(self, e, ei, xi, params):
    """Drawin's formula for ionization."""
    u = e/ei
    alpha, beta = params
    return 1e12*4.0*np.pi*((rB*EI/ei)**2.0)*xi \
      *alpha*(u**(-2.0))*(u-1.0)*np.log(1.25*beta*u)

  def cross_sec_exc(self, e, ei, params):
    """Drawin's formula for excitation."""
    u = e/ei
    alpha, beta, trans_type = params
    if (trans_type == "A"):
      # For optically allowed transitions.
      sigma = ((rB*EI/ei)**2.0)*(u**(-2.0))*(u-1.0)*np.log(1.25*beta*u)
    elif (trans_type == "S"):
      # For spin-forbidden transitions.
      sigma = (rB**2.0)*(u**(-3.0))*(1.0-1.0/(u**2.0))
    elif (trans_type == "P"):
      # For parity-forbidden transitions.
      sigma = (rB**2.0)*(u**(-1.0))*(1.0-1.0/u)
    else:
      pcm.utils.raise_err(
        error=ValueError, text="Unsupported transition type."
      )
    return 1e12*4.0*np.pi*alpha*sigma

  # Reaction rates
  # -----------------------------------
  def eedf(self, e, phi):
    return 3.0*np.sqrt(1.5/np.pi)*phi**(-1.5)*np.exp(-1.5*e/phi)

  def shift_grid(self, e, e_ref, sigma, index):
    """Shift the energy grid by the energy treshold"""
    # Linear interpolation to get the cross-section
    # corresponding to the energy treshold
    sigma_ref = sp.interpolate.interp1d(e, sigma, kind="cubic")(e_ref)
    # Update energy and cross section grids
    e = np.concatenate(([e_ref], e[index:]))
    sigma = np.concatenate(([sigma_ref], sigma[index:]))
    return e, sigma

  def rate(self, e, sigma, phi):
    """
    Function convolving cross-section with electron energy
    distribution function (EEDF) to compute rates.
    """
    # Gamma parameter
    g = np.sqrt(2.0*qe/me)
    # EEDF
    f = self.eedf(e, phi)
    # Integral
    return g*np.trapz(e*sigma*f, e)

  def rate_fwd(self, e, e_ref, sigma, phi):
    """Forward rates"""
    # Shift the energy grid by the energy treshold.
    ne = len(e)
    for i in range(ne):
      if (e_ref < e[i]):
        break
    e, sigma = self.shift_grid(e, e_ref, sigma, i)
    return self.rate(e, sigma, phi)

  def rate_bwd(self, e, e_ref, sigma, phi, gl, gu):
    """Backward rates"""
    # Shift the energy grid by the energy treshold.
    ne = len(e)
    for i in range(ne):
      if (e_ref > (1e-14 + e[i])):
        break
    e, sigma = self.shift_grid(e, e_ref, sigma, i)
    return gl/gu*self.rate(e, sigma, phi)

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
