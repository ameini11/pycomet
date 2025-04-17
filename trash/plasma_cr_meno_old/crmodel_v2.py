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
    self.nb_comp = self.mix["h"].nb_comp - 1
    self.set_rates_values_indices(processes)
    # Corrections
    self.use_corr = use_corr
    self.scale_corr = scale_corr
    if (self.scale_corr is not None):
      self.scale_corr = tf.squeeze(
        tf.constant(self.scale_corr, dtype=bkd.get_floatx())
      )

  def set_rates_values_indices(self, processes):
    # Load processes
    # -------------
    self.processes = processes
    if (not isinstance(self.processes, dict)):
      with open(self.processes, "rb") as file:
        self.processes = pickle.load(file)
    # Collect Arrhenius 'A' matrix
    # -------------
    k = {p: proc["A"] for (p, proc) in self.processes.items()}
    # Get non-zero rates constant
    # -------------
    # > Indices
    indices = tf.nest.map_structure(lambda z: np.where(z > 0.0), k)
    # > Values
    values = {}
    for p in self.processes.keys():
      values[p] = tf.nest.map_structure(
        lambda x: x[indices[p]].reshape(1,-1), self.processes[p]
      )
    # Stacking indices
    indices = {p: np.vstack(indices[p]) for p in self.processes.keys()}
    # Set rates values and indices
    # -------------
    for (k, v, dtype) in (
      ("rates_values", values, bkd.get_floatx()),
      ("rates_indices", indices, "int32")
    ):
      v = tf.nest.map_structure(lambda z: tf.constant(z, dtype=dtype), v)
      setattr(self, k, v)
    # Set scatter updates
    # -------------
    # Matrix of indices
    K = np.arange(self.nb_comp*self.nb_comp)
    K = K.reshape(self.nb_comp,self.nb_comp)
    # Loop over processes
    self.scatter_update = {}
    for p in self.processes.keys():
      self.scatter_update[p] = []
      for (rows, cols, sign, direction) in (
        (0, 0, -1, "fwd"),
        (0, 1,  1, "bwd"),
        (1, 0,  1, "fwd"),
        (1, 1, -1, "bwd")
      ):
        ir = indices[p][rows]
        ic = indices[p][cols]
        ip = K[ir,ic].reshape(-1,1)
        data = {
          "sign": tf.constant(sign, dtype=bkd.get_floatx()),
          "direction": direction,
          "indices": tf.constant(ip, dtype="int32")
        }
        self.scatter_update[p].append(data)
    # Indices for A and b operators
    # -------------
    self.lin_ops_indices = {
      "A": tf.constant(K[1:,1:].reshape(-1), dtype="int32"),
      "b": tf.constant(K[1:,:1].reshape(-1), dtype="int32")
    }

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
    # > Splitting: em, Ar(1-33)
    q = {}
    for k in ("h", "e"):
      q[k] = tf.split(
        value=self.mix[k].q,
        num_or_size_splits=[1, self.nb_comp],
        axis=-1
      )
    # Compute backward rates
    rates["bwd"] = {}
    # > Collisional processes
    for k1 in ("EX", "I"):
      for k2 in ("h", "e"):
        p = k1 + k2
        qf = tf.gather(q[k2][1], indices=self.rates_indices[p][0], axis=-1)
        qb = tf.gather(q[k2][1], indices=self.rates_indices[p][1], axis=-1)
        rates["bwd"][p] = rates["fwd"][p] * qf / qb
        if (k1 == "I"):
          rates["bwd"][p] /= q[k2][0]
    # > Radiation processes
    for p in ("BB", "BF"):
      l = self.rates_values[p]["lambda"]
      rates["bwd"][p] = l * rates["fwd"][p]
      rates["fwd"][p] *= (1.0 - l)
    return rates

  def compose_lin_ops(self, rates, n_em, n_ar1):
    omega = self.compute_sources(rates, n_em, n_ar1)
    m, n = self.nb_comp, tf.shape(n_em)[0]
    Ab = tf.zeros((m*m, n), dtype=bkd.get_floatx())
    for p in self.processes.keys():
      for data in self.scatter_update[p]:
        if data["direction"] == "fwd":
          continue
        Ab = tf.tensor_scatter_nd_add(
          tensor=Ab,
          indices=data["indices"],
          updates=tf.transpose(data["sign"]*omega[data["direction"]][p])
        )
    Ab = tf.transpose(Ab)
    A = tf.reshape(Ab, (-1,m,m))
    return A
    # A = tf.gather(Ab, indices=self.lin_ops_indices["A"], axis=-1)
    # A = tf.reshape(A, (-1,m-1,m-1))
    # b = tf.gather(Ab, indices=self.lin_ops_indices["b"], axis=-1) * n_ar1
    # return A, b

  def compute_sources(self, rates, n_em, n_ar1):
    for z in ("fwd", "bwd"):
      for p in self.processes.keys():
        if p.endswith("e"):
          rates[z][p] *= n_em
        if p.endswith("h"):
          rates[z][p] *= n_ar1
        if ((z == "bwd") and p.startswith("I")):
          rates[z][p] *= n_em
        if ((z == "bwd") and (p == "BF")):
          rates[z][p] *= n_em
    return rates

  # Configuration
  # ===================================
  def get_config(self):
    config = self.mix.get_config()
    config["processes"] = self.processes
    return config
