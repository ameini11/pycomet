import copy
import numpy as np
import dill as pickle
import tensorflow as tf

from . import const
from .mixture import Mixture
from pycomet import backend as bkd


class CRModel(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    processes,
    use_corr=False,
    scale_corr=False,
    species_order=("em", "Ar", "Arp")
  ):
    # Set mixtures
    mix = Mixture(
      species=species,
      species_order=tuple(species_order)
    )
    self.mix = {
      # > Reference mixture
      "h": mix,
      # > Te-based thermo mixture
      "e": copy.deepcopy(mix)
    }
    # Initialize processes rates
    self.nb_comp_nlin = 2
    self.nb_comp_lin = self.mix["h"].nb_comp - self.nb_comp_nlin
    self.set_rates_values_indices(processes)
    # Corrections
    self.use_corr = use_corr
    self.scale_corr = scale_corr
    if (self.scale_corr is not None):
      self.scale_corr = tf.squeeze(
        tf.constant(self.scale_corr, dtype=bkd.get_floatx())
      )

  def set_rates_values_indices(self, processes, eps=0.0):
    # Load processes
    # -------------
    self.processes = processes
    if (not isinstance(self.processes, dict)):
      with open(self.processes, "rb") as file:
        self.processes = pickle.load(file)
    # Collect Arrhenius 'A' matrix
    # -------------
    k = {name: p["A"] for (name, p) in self.processes.items()}
    # Get non-zero rates constant
    # -------------
    # > Indices
    indices_2d = tf.nest.map_structure(
      lambda z: np.where(z > eps), k
    )
    indices_1d = tf.nest.map_structure(
      lambda z: np.where(z.reshape(-1) > eps)[0].reshape(-1,1), k
    )
    # > Values
    values_2d = {}
    for p in self.processes.keys():
      values_2d[p] = tf.nest.map_structure(
        lambda x: x[indices_2d[p]].reshape(1,-1), self.processes[p]
      )
    # Composing
    indices_2d = {p: np.vstack(indices_2d[p]) for p in self.processes}
    # Set rates values and indices
    # -------------
    for (key, value, dtype) in (
      ("rates_values", values_2d, bkd.get_floatx()),
      ("rates_indices", indices_2d, "int32"),
      ("scatter_update_indices", indices_1d, "int32")
    ):
      setattr(self, key, tf.nest.map_structure(
        lambda z: tf.constant(z, dtype=dtype), value
      ))
    self.scatter_size = self.nb_comp_lin**2

  # Rates
  # ===================================
  def compute_rates(self, Th, Te, f):
    Th, Te = [self.clip_temp(z) for z in (Th, Te)]
    # Update species thermo
    self.mix["h"].update_species_thermo(Th, Te)
    self.mix["e"].update_species_thermo(Te, Te)
    # Compute fwd reaction rates
    rates = {"fwd": self.compute_fwd_rates(Th, Te)}
    # Correct fwd reaction rates
    if self.use_corr:
      rates["fwd"] = self.correct_fwd_rates(rates["fwd"], f)
    # Compute bwd reaction rates
    return self.compute_bwd_rates(rates)

  def clip_temp(self, T):
    return tf.clip_by_value(T, const.TMIN, const.TMAX)

  def compute_fwd_rates(self, Th, Te):
    rates = {}
    # > Processes by heavy-particle impact
    ln_T, ov_T = tf.math.log(Th), 1.0/Th
    for p in ("EXh", "Ih"):
      rates[p] = self._arrhenius(ln_T, ov_T, self.rates_values[p])
    # > Processes by electron impact
    ln_T, ov_T = tf.math.log(Te), 1.0/Te
    for p in ("EXe", "Ie", "BF"):
      rates[p] = self._arrhenius(ln_T, ov_T, self.rates_values[p])
    # > Spontaneous emission and absorption (BB)
    rates["BB"] = tf.repeat(
      input=self.rates_values["BB"]["A"],
      repeats=tf.shape(Te)[0],
      axis=0
    )
    return rates

  def _arrhenius(self, ln_T, ov_T, param):
    return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

  def correct_fwd_rates(self, fwd_rates, f):
    # Loop over processes
    for p in self.processes.keys():
      # Compose correction
      pcorr = tf.gather(f, indices=self.rates_indices[p][0], axis=-2) \
            - tf.gather(f, indices=self.rates_indices[p][1], axis=-2)
      # Apply correction
      pcorr = tf.math.reduce_sum(pcorr, axis=-1)
      fwd_rates[p] *= tf.math.exp(pcorr)
    return fwd_rates

  def compute_bwd_rates(self, rates):
    # Extract partition function
    # > Splitting: em, Ar(1), Ar(2-34)
    q = {}
    for k in ("h", "e"):
      q[k] = tf.split(
        value=self.mix[k].q,
        num_or_size_splits=[1, 1, self.nb_comp_lin],
        axis=-1
      )
    # Compute backward rates
    rates["bwd"] = {}
    # > Collisional processes
    for k1 in ("EX", "I"):
      for k2 in ("h", "e"):
        p = k1 + k2
        qf = tf.gather(q[k2][2], indices=self.rates_indices[p][0], axis=-1)
        qb = tf.gather(q[k2][2], indices=self.rates_indices[p][1], axis=-1)
        rates["bwd"][p] = rates["fwd"][p] * qf / qb
        if (k1 == "I"):
          rates["bwd"][p] /= q[k2][0]
    # > Radiation processes
    for p in ("BB", "BF"):
      l = self.rates_values[p]["lambda"]
      rates["bwd"][p] = l * rates["fwd"][p]
      rates["fwd"][p] *= (1.0 - l)
    return rates

  def compose_sources_mat(self, rates, n_em, n_ar1):
    omega = {}
    shape = (self.nb_comp_lin**2, tf.shape(n_em)[0])
    for z in ("fwd", "bwd"):
      omega[z] = tf.zeros(shape, dtype=bkd.get_floatx())
      for p in self.processes.keys():
        omegap = rates[z][p]
        if ("e" in p):
          omegap *= n_em
        if ("h" in p):
          omegap *= n_ar1
        if ((z == "bwd") and ("I" in p)):
          omegap *= n_em
        if ((z == "bwd") and (p == "BF")):
          omegap *= n_em
        omega[z] = tf.tensor_scatter_nd_add(
          tensor=omega[z],
          indices=self.scatter_update_indices[p],
          updates=tf.transpose(omegap)
        )
      omega[z] = tf.reshape(
        tf.transpose(omega[z]), (-1, self.nb_comp_lin, self.nb_comp_lin)
      )
    return omega

  def compose_lin_ops(self, omega):
    A = omega["fwd"] + tf.transpose(omega["bwd"], perm=[0,2,1])
    A = tf.linalg.set_diag(
      input=tf.transpose(A, perm=[0,2,1]),
      diagonal=-tf.math.reduce_sum(A, axis=-1)
    )
    return A

  # Configuration
  # ===================================
  def get_config(self):
    config = self.mix.get_config()
    config["processes"] = self.processes
    return config
