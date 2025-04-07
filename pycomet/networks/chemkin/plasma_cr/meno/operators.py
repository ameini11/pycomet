import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.meno import layers as meno_lay
from silx.io.dictdump import h5todict
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import specie_thermo as st_mod


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NonEqOperators(nn_elem.Layer):
  """Compute the master equation operators for triatomic systems."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    use_corr=True,
    scale_corr=None,
    **kwargs
  ):
    super(NonEqOperators, self).__init__(**kwargs)
    # Reaction rates
    self.nb_comp = None
    self.rates = self.read_rates(rates)
    # Reactive processes
    self.reactions = list(self.rates.keys())
    self.reactions.sort()
    self.processes = []
    for r in self.reactions:
      if (r == "2"):
        self.processes.append("d")
      elif (r in ("5", "6")):
        self.processes.append("e")
      else:
        utils.raise_err(NotImplementedError)
    # Reaction rates values and indices
    self.set_rates_values_indices()
    # Corrections
    self.use_corr = use_corr
    self.scale_corr = scale_corr
    if (self.scale_corr is not None):
      self.scale_corr = tf.squeeze(
        tf.constant(self.scale_corr, dtype=bkd.get_floatx())
      )

  def read_rates(self, rates=None):
    if (rates is not None):
      if isinstance(rates, str):
        rates = h5todict(h5file=rates)
      return tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)

  def set_rates_values_indices(self, eps=1e-50):
    # Assemble Arrhenius 'A' matrix
    # -------------
    A = {p: 0.0 for p in self.processes}
    for r in self.reactions:
      p = self._get_process(r)
      A[p] = A[p] + self.rates[r]["A"]
      # Get number of components
      if (self.nb_comp is None):
        self.nb_comp = A[p].shape[0]
    # Get non-zero rates constant
    # -------------
    # > Indices
    indices_2d = tf.nest.map_structure(
      lambda z: np.where(z > eps), A
    )
    indices_1d = tf.nest.map_structure(
      lambda z: np.where(z.reshape(-1) > eps)[0].reshape(-1,1), A
    )
    # > Values
    values_2d = {}
    for r in self.reactions:
      p = self._get_process(r)
      values_2d[r] = tf.nest.map_structure(
        lambda x: x[indices_2d[p]].reshape(1,-1), self.rates[r]
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
    self.scatter_sizes = {"d": self.nb_comp, "e": self.nb_comp**2}

  def _get_process(self, reaction):
    return "d" if (reaction == "2") else "e"

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > na: atom number density
    # > T:  translational temperature
    # > q:  updated partition functions at temperature T
    # > f:  state-wise corrective factors
    na, T, *q, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(T, q, f)
    # Compose reaction rates matrices
    rates = self.compose_rates_mat(rates, batch_size=tf.shape(T)[0])
    # Compose linear operators: A and b from dydt=A@y+b
    return self.compose_lin_ops(rates, na)

  def compute_rates(self, T, q, f):
    # Compute fwd reaction rates
    rates = {"fwd": self.compute_fwd_rates(T)}
    # Correct fwd reaction rates
    if self.use_corr:
      rates["fwd"] = self.correct_fwd_rates(rates["fwd"], f)
    # Compute bwd reaction rates
    rates["bwd"] = self.compute_bwd_rates(rates["fwd"], q)
    return rates

  def compose_rates_mat(self, rates, batch_size):
    rates_mat = {}
    for z in ("fwd", "bwd"):
      rates_mat[z] = {}
      for p in self.processes:
        rates_mat[z][p] = tf.transpose(tf.scatter_nd(
          indices=self.scatter_update_indices[p],
          updates=tf.transpose(rates[z][p]),
          shape=[self.scatter_sizes[p], batch_size]
        ))
      if ("e" in self.processes):
        rates_mat[z]["e"] = tf.reshape(
          rates_mat[z]["e"], (-1, self.nb_comp, self.nb_comp)
        )
    return rates_mat

  def compose_lin_ops(self, rates, na):
    # > Excitation-Relaxation-Dissociation processes
    A = rates["fwd"]["e"] + tf.transpose(rates["bwd"]["e"], perm=[0,2,1])
    A = tf.linalg.set_diag(
      input=tf.transpose(A, perm=[0,2,1]),
      diagonal=-tf.math.reduce_sum(A, axis=-1)-rates["fwd"]["d"]
    ) * tf.expand_dims(na, axis=-1)
    # > Recombination processes
    b = rates["bwd"]["d"] * na**3
    return A, b

  # Reaction rates
  # -----------------------------------
  def compute_fwd_rates(self, T):
    if (T.shape[1] > 1):
      ln_T, ov_T = [tf.gather(T, indices=[i], axis=-1) for i in (1,2)]
    else:
      ln_T, ov_T = tf.math.log(T), 1.0/T
    rates = {}
    for r in self.reactions:
      rates_r = self._arrhenius(ln_T, ov_T, self.rates_values[r])
      p = self._get_process(r)
      if (p in rates.keys()):
        rates[p] += rates_r
      else:
        rates[p] = rates_r
    return rates

  def _arrhenius(self, ln_T, ov_T, param):
    return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

  def correct_fwd_rates(self, fwd_rates, f):
    # Loop over processes
    for p in self.processes:
      # Compose correction
      pcorr = tf.gather(f, indices=self.rates_indices[p][0], axis=-2)
      if (p == "e"):
        pcorr -= tf.gather(f, indices=self.rates_indices[p][1], axis=-2)
      # Apply correction
      pcorr = tf.math.reduce_sum(pcorr, axis=-1)
      if (self.scale_corr is not None):
        pcorr *= self.scale_corr
      fwd_rates[p] *= tf.math.exp(pcorr)
    return fwd_rates

  def compute_bwd_rates(self, fwd_rates, q):
    # Unpack partition functions (atom, molecule)
    qa, qm = q
    ov_qa_sq = 1.0/qa**2
    # Loop over processes
    bwd_rates = {}
    for p in self.processes:
      qi = tf.gather(qm, indices=self.rates_indices[p][0], axis=-1)
      if (p == "d"):
        # Dissociation processes
        keq = qi * ov_qa_sq
      else:
        # Excitation processes
        keq = qi / tf.gather(qm, indices=self.rates_indices[p][1], axis=-1)
      # Compute value
      bwd_rates[p] = fwd_rates[p] * keq
    return bwd_rates

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": self.rates,
      "use_corr": self.use_corr,
      "scale_corr": ops.to_numpy(self.scale_corr)
    }
    base_config = super(NonEqOperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
