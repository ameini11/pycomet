import tensorflow as tf

from pycomet import utils
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from .utils import read_rates
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
    self.rates = tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.constant(x), axis=0),
      read_rates(rates)
    )
    # Corrections
    self.use_corr = use_corr

    self.processes = ["2","5","6"]
    self.rates_indices = ...
    self.rates_param = ...

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > na: atom number density
    # > Ts: precomputed temperature vectors (T,ln_T,ov_T)
    # > q:  updated partition functions at temperature T
    # > f:  state-wise corrective factors
    na, Ts, q, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(Ts, q, f)
    # Compose reaction rates into matrices
    rates = self.compose_rates(rates)
    # Compute operators
    return self.compute_ops(rates, na)

  def compute_rates(self, Ts, q, f):
    rates = {}
    # Compute fwd reaction rates
    rates["fwd"] = self.compute_fwd_rates(Ts)
    # Correct fwd reaction rates
    if self.use_corr:
      rates["fwd"] = self.apply_corr(rates["fwd"], f)
    # Compute bwd reaction rates
    rates["bwd"] = self.compute_bwd_rates(rates["fwd"], q)
    return rates

  def compose_rates(self, rates):
    return rates

  def compute_ops(self, rates, na):
    # > Excitation-Relaxation-Dissociation processes
    k = tf.linalg.set_diag(
      input=tf.transpose(rates["e"], perm=[0,2,1]),
      diagonal=tf.math.reduce_sum(rates["e"], axis=-1) - rates["d"]
    )
    k *= tf.expand_dims(na, axis=-1)
    # > Recombination processes
    wr = rates["r"] * na**3
    return k, wr

  # Reaction rates corrections
  # -----------------------------------
  def apply_corr(self, fwd_rates, f):
    # Loop over processes
    for k in ("d", "e"):
      if (k in fwd_rates.keys()):
        # Extract correction
        kcorr = tf.gather(f, indices=self.rates["indices"][k][0], axis=-2)
        if (k == "e"):
          kcorr -= tf.gather(f, indices=self.rates["indices"][k][1], axis=-2)
        # Apply correction
        fwd_rates[k] *= tf.math.exp(tf.math.reduce_sum(kcorr, axis=-1))
    return fwd_rates

  # Reaction rates
  # -----------------------------------
  def compute_fwd_rates(self, Ts):
    # Precomputed temperature vectors
    _, ln_T, ov_T = tf.split(Ts, num_or_size_splits=3, axis=-1)
    # Loop over processes
    fwd_rates = {}
    for p in self.processes:
      fwd_rates_p = self._arrhenius(ln_T, ov_T, self.rates["value"][p])
      if (p == "2"):
        # Dissociation processes
        fwd_rates["d"] = fwd_rates_p
      elif (p in ("5", "6")):
        # Excitation processes
        if ("e" in fwd_rates.keys()):
          fwd_rates["e"] += fwd_rates_p
        else:
          fwd_rates["e"] = fwd_rates_p
      else:
        utils.raise_err(NotImplementedError)
    return fwd_rates

  def _arrhenius(self, ln_T, ov_T, param):
    return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

  def compute_bwd_rates(self, fwd_rates, q):
    # Unpack partition functions (atom, molecule)
    qa, qm = tf.split(q, num_or_size_splits=[1,q.shape[1]-1], axis=-1)
    # Loop over processes
    bwd_rates = {}
    for k in ("d", "e"):
      if (k in fwd_rates.keys()):
        qi = tf.gather(qm, indices=self.rates["indices"][k][0], axis=-1)
        if (k == "d"):
          # Dissociation processes
          ov_qj = 1/(qa**2)
        else:
          # Excitation processes
          ov_qj = 1/tf.gather(qm, indices=self.rates["indices"][k][1], axis=-1)
        # Compute value
        bwd_rates[k] = fwd_rates[k] * qi * ov_qj
    return bwd_rates

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_corr": self.use_corr
    }
    base_config = super(Operators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
