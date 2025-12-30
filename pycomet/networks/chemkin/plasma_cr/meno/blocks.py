import copy
import numpy as np
import tensorflow as tf

from . import const
from .mixture import Mixture
from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.meno import layers as meno_lay
from silx.io.dictdump import h5todict
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCROperators(nn_elem.Layer):
  """Compute the linear operators for thecollisional-radiative Argon
  plasma reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    rates,
    use_corr=False,
    scale_corr=False,
    use_mass_frac=False,
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
    self.nb_comp = self.mix["h"].nb_comp - 1
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
    # State vector
    self.use_mass_frac = bool(use_mass_frac)

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
    # Compose linear operators: A from 'dydt = A @ y'
    A = self.compose_lin_ops(omega)
    # Convert linear operators: number densities to mass fractions
    if self.use_mass_frac:
      A = self.convert_lin_ops(A)
    return A

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
        num_or_size_splits=[1, self.nb_comp],
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
    m, n = self.nb_comp, tf.shape(n_em)[0]
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

  def compose_lin_ops(self, omega):
    # Merged linear-constant term
    A = omega["fwd"] + tf.transpose(omega["bwd"], perm=[0,2,1])
    A = tf.linalg.set_diag(
      input=tf.transpose(A, perm=[0,2,1]),
      diagonal=-tf.math.reduce_sum(A, axis=-1)
    )
    return tf.reshape(A, (-1,self.nb_comp,self.nb_comp))

  def convert_lin_ops(self, A):
    # Convert linear operators from number densities to mass fractions state vector
    # Equation: Aw = M @ An @ M_inv
    A = tf.transpose(A*self.mix["h"].mh_inv, perm=(0,2,1))
    A = tf.transpose(A*self.mix["h"].mh, perm=(0,2,1))
    return A

  # Configuration
  # ===================================
  def get_config(self):
    config = self.mix["h"].get_config()
    config["rates"] = self.rates
    config["use_corr"] = self.use_corr
    config["scale_corr"] = ops.to_numpy(self.scale_corr)
    config["use_mass_frac"] = self.use_mass_frac
    base_config = super(PlasmaCROperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRMoleMass(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    **kwargs
  ):
    super(PlasmaCRMoleMass, self).__init__(**kwargs)
    # Set mixture
    self.mix = Mixture(species)

  # Calling
  # ===================================
  def call(self, inputs, method=3, training=None):
    if (method == 1):
      # Convert from 'rho' to 'w' (mass fractions)
      rho = tf.math.reduce_sum(inputs, axis=-1, keepdims=True)
      return rho, inputs/rho
    elif (method == 2):
      # Convert from 'n' to 'rho'
      return self.mix.get_rho(inputs)
    elif (method == 3):
      # Convert from 'rho' to 'n'
      return self.mix.get_n(inputs)
    elif (method == 4):
      # Convert from 'n' to 'w' (mass fractions)
      rho = self.mix.get_rho(inputs)
      return rho / tf.math.reduce_sum(rho, axis=-1, keepdims=True)
    else:
      utils.raise_err(
        error=ValueError,
        text="Invalid method: it must be an integer between 1 and 4."
      )

  # Configuration update
  # ===================================
  def get_config(self):
    config = self.mix.get_config()
    base_config = super(PlasmaCRMoleMass, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRSolUpdate(nn_elem.Layer):
  """Matrix exponential-based update for the collisional-radiative Argon
  plasma reactor."""

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
    # Ions indices
    self.ii = self.operators.mix["h"].species["Arp"].indices - 1

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, nh_0, u, f = inputs
    # Compute operators
    A = self.operators((*u, f), training=training)
    # Compute number densities
    # > Argon states
    nh = self.linode((t, nh_0, A), training=training)
    # > Electron (charge conservation)
    n = self.enforce_charge_cons(nh)
    return n

  def enforce_charge_cons(self, nh):
    # Compute electron number density
    ne = tf.gather(nh, self.ii, axis=-1)
    ne = tf.math.reduce_sum(ne, axis=-1, keepdims=True)
    return tf.concat([ne, nh], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      **self.operators.get_config(),
      **self.linode.get_config()
    }
    base_config = super(PlasmaCRSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
