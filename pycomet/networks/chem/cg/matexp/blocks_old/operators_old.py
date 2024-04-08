import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from .utils import read_rates
_CFG = get_config()


# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class Operators(nn_elem.Layer):
#   """Compute the master equation operators for triatomic systems."""

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     rates=None,
#     use_corr=True,
#     **kwargs
#   ):
#     super(Operators, self).__init__(**kwargs)
#     # Reaction rates
#     self.rates = tf.nest.map_structure(
#       lambda x: tf.expand_dims(tf.constant(x), axis=0),
#       read_rates(rates)
#     )
#     # Corrections
#     self.use_corr = use_corr

#     self.reactions = ["2","5","6"]
#     self.rates_indices = ...
#     self.rates_param = ...

#   # Calling
#   # ===================================
#   def call(self, inputs, training=None):
#     # Extract inputs:
#     # > na: atom number density
#     # > Ts: precomputed temperature vectors (T,ln_T,ov_T)
#     # > q:  updated partition functions at temperature T
#     # > f:  state-wise corrective factors
#     na, Ts, q, f = inputs
#     # Compute reaction rates
#     rates = self.compute_rates(Ts, q, f)
#     # Compose reaction rates into matrices
#     rates = self.compose_rates(rates)
#     # Compute operators
#     return self.compute_ops(rates, na)

#   def compute_rates(self, Ts, q, f):
#     # Compute fwd reaction rates
#     rates = {"fwd": self.compute_fwd_rates(Ts)}
#     # Correct fwd reaction rates
#     if self.use_corr:
#       rates["fwd"] = self.apply_corr(rates["fwd"], f)
#     # Compute bwd reaction rates
#     rates["bwd"] = self.compute_bwd_rates(rates["fwd"], q)
#     return rates

#   def compose_rates(self, rates):
#     return rates

#   def compute_ops(self, rates, na):
#     # > Excitation-Relaxation-Dissociation processes
#     k = tf.linalg.set_diag(
#       input=tf.transpose(rates["e"], perm=[0,2,1]),
#       diagonal=tf.math.reduce_sum(rates["e"], axis=-1) - rates["d"]
#     ) * tf.expand_dims(na, axis=-1)
#     # > Recombination processes
#     wr = rates["r"] * na**3
#     return k, wr

#   # Reaction rates corrections
#   # -----------------------------------
#   def apply_corr(self, fwd_rates, f):
#     # Loop over processes
#     for k in ("d", "e"):
#       if (k in fwd_rates.keys()):
#         # Extract correction
#         kcorr = tf.gather(f, indices=self.rates["indices"][k][0], axis=-2)
#         if (k == "e"):
#           kcorr -= tf.gather(f, indices=self.rates["indices"][k][1], axis=-2)
#         # Apply correction
#         fwd_rates[k] *= tf.math.exp(tf.math.reduce_sum(kcorr, axis=-1))
#     return fwd_rates

#   # Reaction rates
#   # -----------------------------------
#   def compute_fwd_rates(self, Ts):
#     # Precomputed temperature vectors
#     _, ln_T, ov_T = tf.split(Ts, num_or_size_splits=3, axis=-1)
#     # Loop over processes
#     fwd_rates = {}
#     for r in self.reactions:
#       fwd_rates_p = self._arrhenius(ln_T, ov_T, self.rates["value"][p])
#       if (r == "2"):
#         # Dissociation processes
#         fwd_rates["d"] = fwd_rates_p
#       elif (r in ("5", "6")):
#         # Excitation processes
#         if ("e" in fwd_rates.keys()):
#           fwd_rates["e"] += fwd_rates_p
#         else:
#           fwd_rates["e"] = fwd_rates_p
#       else:
#         utils.raise_err(NotImplementedError)
#     return fwd_rates

#   def _arrhenius(self, ln_T, ov_T, param):
#     return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

#   def compute_bwd_rates(self, fwd_rates, q):
#     # Unpack partition functions (atom, molecule)
#     qa, qm = tf.split(q, num_or_size_splits=[1,q.shape[1]-1], axis=-1)
#     ov_qa_sq = 1/qa**2
#     # Loop over processes
#     bwd_rates = {}
#     for k in ("d", "e"):
#       if (k in fwd_rates.keys()):
#         qi = tf.gather(qm, indices=self.rates["indices"][k][0], axis=-1)
#         if (k == "d"):
#           # Dissociation processes
#           qij = qi * ov_qa_sq
#         else:
#           # Excitation processes
#           qij = qi/tf.gather(qm, indices=self.rates["indices"][k][1], axis=-1)
#         # Compute value
#         bwd_rates[k] = fwd_rates[k] * qij
#     return bwd_rates

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
#       "use_corr": self.use_corr
#     }
#     base_config = super(Operators, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))



















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
      lambda x: ops.to_numpy(x).squeeze(), rates
    )
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

  def get_rates_values_indices(self):
    # Assemble Arrhenius 'A' matrix
    a = {k: 0.0 for k in self.processes}
    for r in self.reactions:
      p = self._get_process(r)
      a[p] = a[p] + self.rates_values[r]["A"]
    # Get non-zero indices
    indices = tf.nest.map_structure(
      lambda z: np.where(z.squeeze() > 0), a
    )
    indices_1d = tf.nest.map_structure(
      lambda z: np.argwhere(z.reshape(-1) > 0)[0].reshape(-1,1), a
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
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_corr": self.use_corr
    }
    base_config = super(Operators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



























# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class Rates(nn_elem.Layer):

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     processes=[],
#     values=None,
#     indices=None,
#     use_corr=True,
#     **kwargs
#   ):
#     super(Rates, self).__init__(**kwargs)
#     # Reactive processes
#     # -------------
#     self.reactions = list(processes).sort()
#     self.processes = []
#     for r in self.reactions:
#       if (r == "2"):
#         self.processes.append("d")
#       elif (r in ("5", "6")):
#         self.processes.append("e")
#       else:
#         utils.raise_err(NotImplementedError)
#     # Reaction rates
#     # -------------
#     # > Get values and indices
#     self.values = tf.nest.map_structure(
#       lambda x: tf.reshape(tf.constant(x, dtype=bkd.floatx()), (1,-1)),
#       values
#     )
#     self.indices = tf.nest.map_structure(
#       lambda x: tf.squeeze(tf.constant(x, dtype="int32")),
#       indices
#     )
#     # > Convert rates
#     self.rates = tf.nest.map_structure(tf.constant, self.rates)
#     for (k, dtype) in (("values", bkd.floatx()), ("indices", "int32")):
#       self.rates[k] = tf.nest.map_structure(
#         lambda x: tf.cast(x, dtype=bkd.floatx()), self.rates[k]
#       )
#     # Corrections
#     # -------------
#     self.use_corr = use_corr

#   def get_rates_indices(self):
#     # Get non-zero indices
#     a = {k: 0.0 for k in self.processes}
#     for r in self.reactions:
#       k = "d" if (r == "2") else "e"
#       a[k] = a[k] + self.rates_values[p]["A"]
#     indices = tf.nest.map_structure(lambda x: np.where(x>0), a)
#     # Extract non-zero rates constant
#     rates = {}
#     for r in self.reactions:
#       k = "d" if (r == "2") else "e"
#       rates[p] = tf.nest.map_structure(
#         lambda x: x[indices["d"]], self.rates_values[p]
#       )


#   def get_rates_indices(self, x):
#     shape = np.shape(x)
#     if ((len(shape) != 2) or (2 not in shape)):
#       utils.raise_err(
#         ValueError,
#         f"'indices' input must be a 2-dimensional " \
#         "array with a shape of [nb_rates, 2]."
#       )
#     if (shape[1] == 2):
#       x = np.transpose(x)
#     return tf.constant(x, dtype=tf.int32)

#   # Calling
#   # ===================================
#   def call(self, inputs, training=None):
#     # Extract inputs:
#     # > T: translational temperature
#     # > f: state-wise corrective factors
#     T, f = inputs
#     # Compute fwd reaction rates
#     rates = self._compute_fwd(T)
#     # Correct fwd reaction rates
#     if self.use_corr:
#       rates = self._correct_fwd(rates, f)
#     # Compute bwd reaction rates
#     rates = self._compute_bwd(T)
#     return [rates[k] for k in self.processes]

#   def _compute_fwd(self, T):
#     ln_T, ov_T = tf.math.log(T), 1/T
#     # Loop over processes
#     rates = {}
#     for r in self.reactions:
#       rates_p = self._arrhenius(ln_T, ov_T, self.rates["value"][p])
#       if (r == "2"):
#         # Dissociation processes
#         rates["d"] = rates_p
#       else:
#         # Excitation processes
#         if ("e" in rates.keys()):
#           rates["e"] += rates_p
#         else:
#           rates["e"] = rates_p
#     return rates

#   def _arrhenius(self, ln_T, ov_T, param):
#     return param["A"] * tf.math.exp(param["beta"]*ln_T - param["Ta"]*ov_T)

#   def _correct_fwd(self, rates, f):
#     # Loop over processes
#     for k in self.processes:
#       # Compose correction
#       kcorr = tf.gather(f, indices=self.rates["indices"][k][0], axis=-2)
#       if (k == "e"):
#         kcorr -= tf.gather(f, indices=self.rates["indices"][k][1], axis=-2)
#       # Apply correction
#       rates[k] *= tf.math.exp(tf.math.reduce_sum(kcorr, axis=-1))
#     return rates

#   def _compute_bwd(self, fwd_rates, q):
#     # Unpack partition functions (atom, molecule)
#     qa, qm = tf.split(q, num_or_size_splits=[1,q.shape[1]-1], axis=-1)
#     ov_qa_sq = 1/qa**2
#     # Loop over processes
#     bwd_rates = {}
#     for k in ("d", "e"):
#       if (k in fwd_rates.keys()):
#         qi = tf.gather(qm, indices=self.rates["indices"][k][0], axis=-1)
#         if (k == "d"):
#           # Dissociation processes
#           qij = qi * ov_qa_sq
#         else:
#           # Excitation processes
#           qij = qi/tf.gather(qm, indices=self.rates["indices"][k][1], axis=-1)
#         # Compute value
#         bwd_rates[k] = fwd_rates[k] * qij
#     return bwd_rates

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       "processes": self.reactions,
#       "rates_values": self.rates_values,
#       "use_corr": self.use_corr
#     }
#     base_config = super(Rates, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))
