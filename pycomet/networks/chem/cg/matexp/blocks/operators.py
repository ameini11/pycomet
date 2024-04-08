import numpy as np
import tensorflow as tf

from pycomet import utils
from .utils import read_rates
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Operators(nn_elem.Layer):
  """Compute the master equation operators for triatomic systems."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    use_corr=True,
    **kwargs
  ):
    super(Operators, self).__init__(**kwargs)
    # Reaction rates
    self.rates = read_rates(rates)
    # Reactive processes
    self.reactions = list(self.rates.keys()).sort()
    self.processes = []
    for r in self.reactions:
      if (r == "2"):
        self.processes.append("d")
      elif (r in ("5", "6")):
        self.processes.append("e")
      else:
        utils.raise_err(NotImplementedError)
    # Reaction rates values and indices
    rates_vi = tf.nest.map_structure(
      tf.constant, self.get_rates_values_indices()
    )
    self.rates_values = tf.cast(rates_vi[0], dtype=bkd.floatx())
    self.rates_indices = tf.cast(rates_vi[1], dtype="int32")
    self.scatter_indices = tf.cast(rates_vi[2], dtype="int32")
    # Corrections
    self.use_corr = use_corr

  def get_rates_values_indices(self, eps=1e-50):
    # Assemble Arrhenius 'A' matrix
    a = {k: 0.0 for k in self.processes}
    for r in self.reactions:
      p = self._get_process(r)
      a[p] = a[p] + self.rates[r]["A"]
    # Get non-zero indices
    indices = tf.nest.map_structure(lambda z: np.where(z > eps), a)
    indices_1d = tf.nest.map_structure(
      lambda z: np.where(z.reshape(-1) > eps)[0].reshape(-1,1), a
    )
    # Get non-zero rates constant
    values = {}
    for r in self.reactions:
      p = self._get_process(r)
      values[r] = tf.nest.map_structure(
        lambda x: x[indices[p]].reshape(1,-1), self.rates[r]
      )
    return values, np.vstack(indices), indices_1d

  def _get_process(reaction):
    return "d" if (reaction == "2") else "e"

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > na: atom number density
    # > T:  translational temperature
    # > q:  updated partition functions at temperature T
    # > f:  state-wise corrective factors
    na, T, q, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(T, q, f)
    # Compose reaction rates
    rates = self.compose_rates(rates, batch_size=tf.shape(T)[0])
    # Compose operators
    return self.compose_ops(rates, na)

  def compute_rates(self, T, q, f):
    # Compute fwd reaction rates
    rates = {"fwd": self.compute_fwd_rates(T)}
    # Correct fwd reaction rates
    if self.use_corr:
      rates["fwd"] = self.correct_fwd_rates(rates["fwd"], f)
    # Compute bwd reaction rates
    rates["bwd"] = self.compute_bwd_rates(rates["fwd"], q)
    return rates

  def compose_rates(self, rates, batch_size):
    for k in self.processes:
      size = self.nb_comp if (k == "d") else self.nb_comp**2
      for z in ("fwd", "bwd"):
        rates_zk = tf.transpose(rates[z][k])
        rates_zk = tf.scatter_nd(
          indices=self.scatter_indices[k],
          updates=rates_zk,
          shape=[size, batch_size]
        )
        rates_zk = tf.transpose(rates_zk)
        if (k == "e"):
          rates_zk = tf.reshape(rates_zk, (-1, self.nb_comp, self.nb_comp))
        rates[z][k] = rates_zk
    return rates

  def compose_ops(self, rates, na):
    # > Excitation-Relaxation-Dissociation processes
    k = rates["fwd"]["e"] + tf.transpose(rates["bwd"]["e"], perm=[0,2,1])
    k = tf.linalg.set_diag(
      input=tf.transpose(k, perm=[0,2,1]),
      diagonal=tf.math.reduce_sum(k, axis=-1) - rates["fwd"]["d"]
    ) * tf.expand_dims(na, axis=-1)
    # > Recombination processes
    w = rates["bwd"]["d"] * na**3
    return k, w

  # Reaction rates
  # -----------------------------------
  def compute_fwd_rates(self, T):
    ln_T, ov_T = tf.math.log(T), 1/T
    # Loop over processes
    rates = {}
    for r in self.reactions:
      rates_p = self._arrhenius(ln_T, ov_T, self.rates_values[p])
      if (r == "2"):
        # Dissociation processes
        rates["d"] = rates_p
      else:
        # Excitation processes
        if ("e" in rates.keys()):
          rates["e"] += rates_p
        else:
          rates["e"] = rates_p
    return rates

  def _arrhenius(self, ln_T, ov_T, param):
    return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

  def correct_fwd_rates(self, fwd_rates, f):
    # Loop over processes
    for k in self.processes:
      # Compose correction
      kcorr = tf.gather(f, indices=self.rates_indices[k][0], axis=-2)
      if (k == "e"):
        kcorr -= tf.gather(f, indices=self.rates_indices[k][1], axis=-2)
      # Apply correction
      fwd_rates[k] *= tf.math.exp(tf.math.reduce_sum(kcorr, axis=-1))
    return fwd_rates

  def compute_bwd_rates(self, fwd_rates, q):
    # Unpack partition functions (atom, molecule)
    qa, qm = tf.split(q, num_or_size_splits=[1,q.shape[1]-1], axis=-1)
    ov_qa_sq = 1/qa**2
    # Loop over processes
    bwd_rates = {}
    for k in self.processes:
      qi = tf.gather(qm, indices=self.rates_indices[k][0], axis=-1)
      if (k == "d"):
        # Dissociation processes
        keq = qi * ov_qa_sq
      else:
        # Excitation processes
        keq = qi / tf.gather(qm, indices=self.rates_indices[k][1], axis=-1)
      # Compute value
      bwd_rates[k] = fwd_rates[k] * keq
    return bwd_rates

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": self.rates,
      "use_corr": self.use_corr
    }
    base_config = super(Operators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
