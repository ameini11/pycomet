import copy
import numpy as np
import tensorflow as tf

from . import const
from .mixture import Mixture
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.meno import layers as meno_lay
from silx.io.dictdump import h5todict
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCROperators(nn_elem.Layer):
  """Compute the linear operators for thecollisional-radiative Argon plasma reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    rates,
    use_corr=False,
    scale_corr=False,
    **kwargs
  ):
    super(PlasmaCROperators, self).__init__(**kwargs)
    # Set mixtures
    mix = Mixture(species)
    self.mix = {
      # > Reference mixture
      "h": mix,
      # > Te-based thermo mixture
      "e": copy.deepcopy(mix)
    }
    self.nb_comp = self.mix["h"].nb_comp - 2
    # Initialize rates
    self.rates = self.read_rates(rates)
    self.processes = list(self.rates.keys())
    self.set_rates_values_indices()
    # Corrections
    self.use_corr = use_corr
    self.scale_corr = scale_corr
    if (self.scale_corr is not None):
      self.scale_corr = tf.squeeze(
        tf.constant(self.scale_corr, dtype=bkd.get_floatx())
      )

  def read_rates(self, rates):
    if isinstance(rates, str):
      rates = h5todict(h5file=rates)
    return tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)

  def set_rates_values_indices(self):
    # Collect Arrhenius 'A' matrix
    # -------------
    a = {p: self.rates[p]["A"] for p in self.processes}
    # Get non-zero rates constant
    # -------------
    # > Indices
    indices_2d = tf.nest.map_structure(
      lambda z: np.where(z > 0.0), a
    )
    indices_1d = tf.nest.map_structure(
      lambda z: np.where(z.reshape(-1) > 0.0)[0].reshape(-1,1), a
    )
    # > Values
    values_2d = {}
    for p in self.processes:
      values_2d[p] = tf.nest.map_structure(
        lambda x: x[indices_2d[p]].reshape(1,-1), self.rates[p]
      )
    # Composing indices
    indices_2d = {p: np.vstack(indices_2d[p]) for p in self.rates}
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
    # Set indices for A and b operators
    # -------------
    # Matrix of indices
    m = self.nb_comp + 1
    K = np.arange(m*m).reshape(m,m)
    # Exclude ground state of Argon
    self.lin_ops_indices = {
      "A": tf.constant(K[1:,1:].reshape(-1), dtype="int32"),
      "b": tf.constant(K[1:,:1].reshape(-1), dtype="int32")
    }

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > n_em:  electron number density
    # > n_ar1: argon ground state number density
    # > Th:    heavy-particle temperature
    # > Te:    electron temperature
    # > f:     state-wise corrective factors
    n_em, n_ar1, Th, Te, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(Th, Te, f)
    # Compose reaction rates matrices
    omega = self.compose_sources_mat(rates, n_em, n_ar1)
    # Compose linear operators: A and b from dydt=A@y+b
    return self.compose_lin_ops(omega, n_ar1)

  # Rates
  # -----------------------------------
  def compute_rates(self, Th, Te, f):
    Th, Te = [self._clip_temp(z) for z in (Th, Te)]
    # Update species thermo
    self.mix["h"].update_species_thermo(Th, Te)
    self.mix["e"].update_species_thermo(Te, Te)
    # Compute fwd reaction rates
    rates = {"fwd": self._compute_fwd_rates(Th, Te)}
    # Correct fwd reaction rates
    if self.use_corr:
      rates["fwd"] = self._correct_fwd_rates(rates["fwd"], f)
    # Compute bwd reaction rates
    return self._compute_bwd_rates(rates)

  def _clip_temp(self, T):
    return tf.clip_by_value(T, const.TMIN, const.TMAX)

  def _compute_fwd_rates(self, Th, Te):
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

  def _correct_fwd_rates(self, fwd_rates, f):
    # Insert Argon ground state correction
    f1 = tf.zeros_like(tf.gather(f, indices=[0], axis=-2))
    f = tf.concat([f1,f], axis=-2)
    # Loop over rates
    for p in self.processes:
      # Compose correction
      pcorr = tf.gather(f, indices=self.rates_indices[p][0], axis=-2) \
            - tf.gather(f, indices=self.rates_indices[p][1], axis=-2)
      # Apply correction
      pcorr = tf.math.reduce_sum(pcorr, axis=-1)
      fwd_rates[p] *= tf.math.exp(pcorr)
    return fwd_rates

  def _compute_bwd_rates(self, rates):
    # Extract partition function
    # > Splitting: em, Ar(1-33)
    q = {}
    for a in ("h", "e"):
      q[a] = tf.split(
        value=self.mix[a].q,
        num_or_size_splits=[1, self.nb_comp+1],
        axis=-1
      )
    # Compute backward rates
    rates["bwd"] = {}
    # > Collisional rates
    for k1 in ("EX", "I"):
      for k2 in ("h", "e"):
        p = k1 + k2
        qf = tf.gather(q[k2][1], indices=self.rates_indices[p][0], axis=-1)
        qb = tf.gather(q[k2][1], indices=self.rates_indices[p][1], axis=-1)
        rates["bwd"][p] = rates["fwd"][p] * qf / qb
        if (k1 == "I"):
          rates["bwd"][p] /= q[k2][0]
    # > Radiation rates
    for p in ("BB", "BF"):
      l = self.rates_values[p]["lambda"]
      rates["bwd"][p] = l * rates["fwd"][p]
      rates["fwd"][p] *= (1.0 - l)
    return rates

  # Operators
  # -----------------------------------
  def compose_sources_mat(self, rates, n_em, n_ar1):
    omega = {}
    m, n = self.nb_comp + 1, tf.shape(n_em)[0]
    for z in ("fwd", "bwd"):
      omega[z] = tf.zeros((m*m, n), dtype=bkd.get_floatx())
      for p in self.processes:
        if p.endswith("e"):
          rates[z][p] *= n_em
        if p.endswith("h"):
          rates[z][p] *= n_ar1
        if ((z == "bwd") and p.startswith("I")):
          rates[z][p] *= n_em
        if ((z == "bwd") and (p == "BF")):
          rates[z][p] *= n_em
        omega[z] = tf.tensor_scatter_nd_add(
          tensor=omega[z],
          indices=self.scatter_update_indices[p],
          updates=tf.transpose(rates[z][p])
        )
      omega[z] = tf.reshape(tf.transpose(omega[z]), (-1,m,m))
    return omega

  def compose_lin_ops(self, omega, n_ar1):
    m = self.nb_comp + 1
    # Merged linear-constant term
    Ab = omega["fwd"] + tf.transpose(omega["bwd"], perm=[0,2,1])
    Ab = tf.linalg.set_diag(
      input=tf.transpose(Ab, perm=[0,2,1]),
      diagonal=-tf.math.reduce_sum(Ab, axis=-1)
    )
    Ab = tf.reshape(Ab, (-1,m*m))
    # Linear term
    A = tf.gather(Ab, indices=self.lin_ops_indices["A"], axis=-1)
    A = tf.reshape(A, (-1,m-1,m-1))
    # Constant term
    b = tf.gather(Ab, indices=self.lin_ops_indices["b"], axis=-1) * n_ar1
    return A, b

  # Configuration
  # ===================================
  def get_config(self):
    config = self.mix["h"].get_config()
    config["rates"] = self.rates
    config["use_corr"] = self.use_corr
    config["scale_corr"] = ops.to_numpy(self.scale_corr)
    base_config = super(PlasmaCROperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRMoleMass(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    charge_cons=False,
    **kwargs
  ):
    super(PlasmaCRMoleMass, self).__init__(**kwargs)
    # Set mixture
    self.mix = Mixture(species)
    self.charge_cons = bool(charge_cons)

  # Calling
  # ===================================
  def call(self, inputs, method=3, training=None):
    if (method == 1):
      # Convert from 'rho' to 'w' (mass fractions)
      rho = tf.math.reduce_sum(inputs, axis=-1, keepdims=True)
      return rho, inputs/rho
    elif (method == 2):
      # Convert from 'n' to 'rho'
      if self.charge_cons:
        inputs = self.enforce_charge_cons(inputs)
      return self.mix.get_rho(inputs)
    elif (method == 3):
      # Convert from 'rho' to 'n'
      return self.mix.get_n(inputs)
    elif (method == 4):
      # Convert from 'n' to 'w' (mass fractions)
      if self.charge_cons:
        inputs = self.enforce_charge_cons(inputs)
      rho = self.mix.get_rho(inputs)
      return rho / tf.math.reduce_sum(rho, axis=-1, keepdims=True)

  def enforce_charge_cons(self, n):
    # Get indices
    ii, ie = [self.mix.species[k].indices for k in ("Arp", "em")]
    # Compute ions number density
    ni = tf.gather(n, ii, axis=-1)
    ni = tf.math.reduce_sum(ni, axis=-1, keepdims=True)
    # Set electron number density
    return tf.transpose(tf.tensor_scatter_nd_update(
      tensor=tf.transpose(n),
      indices=tf.reshape(ie, (-1,1)),
      updates=tf.transpose(ni)
    ))

  # Configuration update
  # ===================================
  def get_config(self):
    config = self.mix.get_config()
    config["charge_cons"] = self.charge_cons
    base_config = super(PlasmaCRMoleMass, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRSolUpdate(nn_elem.Layer):
  """Matrix exponential-based update for the collisional-radiative Argon plasma reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    rates,
    use_corr=False,
    scale_corr=False,
    epsilon=1e-15,
    use_pade=False,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(PlasmaCRSolUpdate, self).__init__(**kwargs)
    # PlasmaCROperators layer
    self.operators = PlasmaCROperators(
      species=species,
      rates=rates,
      use_corr=use_corr,
      scale_corr=scale_corr
    )
    # LinODE Layer
    self.linode = meno_lay.LinODE(
      nb_eqs=self.operators.nb_comp,
      epsilon=epsilon,
      use_pade=use_pade,
      use_expmv=use_expmv,
      max_degree=max_degree,
      max_nb_squarings=max_nb_squarings,
      expm_dtype=expm_dtype,
      add_inv_loss=add_inv_loss,
      inv_loss_weight=inv_loss_weight
    )

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, xlin_0, xnlin, f = inputs
    # Compute operators
    A, b = self.operators((*xnlin, f), training=training)
    # Compute number densities of the 'linear' components
    return self.linode((t, xlin_0, A, b), training=training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      **self.operators.get_config(),
      **self.linode.get_config()
    }
    base_config = super(PlasmaCRSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
