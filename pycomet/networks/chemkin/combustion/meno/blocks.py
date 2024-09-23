import numpy as np
import tensorflow as tf

from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.meno import layers as meno_lay
from silx.io.dictdump import h5todict
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class POLLUOperators(nn_elem.Layer):
  """Compute the linear operators for POLLU reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    use_corr=True,
    **kwargs
  ):
    super(POLLUOperators, self).__init__(**kwargs)
    # Reaction rates
    self.rates = self.read_rates(rates)
    self.set_rates_values_indices()
    # Corrections
    self.use_corr = use_corr

  # def read_rates(self, rates):
  #   if isinstance(rates, str):
  #     rates = h5todict(h5file=rates)
  #   rates = tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)
  #   rates["indices"] = rates["indices"].astype(np.int32)
  #   return rates

  # def set_rates_values_indices(self):
  #   self.nb_comp = int(np.amax(self.rates["indices"]))+1
  #   # Set scatter update
  #   self.scatter_update = []
  #   i = self.rates["indices"]
  #   K = np.arange(self.nb_comp**2).reshape(self.nb_comp,self.nb_comp)
  #   for (rows, cols, sign, direction) in (
  #     (0, 0, -1, "fwd"),
  #     (0, 1,  1, "bwd"),
  #     (1, 0,  1, "fwd"),
  #     (1, 1, -1, "bwd")
  #   ):
  #     indices = K[i[rows],i[cols]].reshape(-1,1)
  #     data = {
  #       "sign": tf.constant(sign, dtype=bkd.get_floatx()),
  #       "direction": direction,
  #       "indices": tf.constant(indices, dtype="int32")
  #     }
  #     self.scatter_update.append(data)
  #   # Set rates values and indices
  #   self.rates_values = {}
  #   for direction in ("fwd", "bwd"):
  #     self.rates_values[direction] = tf.constant(
  #       self.rates[direction].reshape(1,-1), dtype=bkd.get_floatx()
  #     )
  #   self.rates_indices = tf.constant(self.rates["indices"], dtype="int32")

  # def _set_amat_indices(self):
  #   s_l = self.species_split["lin"]
  #   nb_eqs = len(s_l)
  #   ireac, iprod, creac, cprod = [], [], [], []
  #   for r in range(self.nb_reactions):
  #     # Species indices and stoichiometric coefficients
  #     # > Reactants
  #     ir_1d, cr_1d = [self.rates["reac"][k][r] for k in ("i", "c")]
  #     # > Products
  #     ip_1d, cp_1d = [self.rates["prod"][k][r] for k in ("i", "c")]
  #     if (np.sum(ir_1d[s_l]) == 0):
  #       ir = cr = ip = cp = np.zeros(nb_eqs**2)
  #     else:
  #       ir = np.outer(ir_1d[s_l],ir_1d[s_l]).reshape(-1)
  #       ip = np.outer(ip_1d[s_l],ir_1d[s_l]).reshape(-1)
  #       cr = np.outer(cr_1d[s_l],ir_1d[s_l]).reshape(-1)
  #       cp = np.outer(cp_1d[s_l],ir_1d[s_l]).reshape(-1)
  #     ireac.append(ir)
  #     iprod.append(ip)
  #     creac.append(cr)
  #     cprod.append(cp)
  #   return {
  #     "reac": {
  #       "i": np.vstack(ireac).astype(bool),
  #       "c": np.vstack(creac)
  #     },
  #     "prod": {
  #       "i": np.vstack(iprod).astype(bool),
  #       "c": np.vstack(cprod)
  #     }
  #   }

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > y_nl: nonlinear state variables
    # > f:    state-wise corrective factors
    y_nl, f = inputs
    # Compose linear operators: A from dydt=A@y
    batch_size = tf.shape(y_nl)[0]
    a = tf.zeros((self.nb_comp**2, batch_size), dtype=bkd.get_floatx())
    b = tf.zeros((self.nb_comp,    batch_size), dtype=bkd.get_floatx())
    s_l, s_nl = [self.species_split[k] for k in ("lin", "nlin")]
    # Loop over reactions
    for r in tf.range(self.nb_reactions):
      # Reactants indices
      i = self.rates_indices["reac"]["i"][r]
      # Source term
      w = self.compute_source_term(y_nl, f, r, i[s_nl], batch_size)
      # Update linear operators
      if (tf.math.reduce_sum(i[s_l]) == 0):
        b = self.update_bvec(b, r, w, s_l)
      else:
        a = self.update_amat(a, r, w)
    A = tf.reshape(tf.transpose(a), (-1,self.nb_comp,self.nb_comp))
    b = tf.transpose(b)
    return A, b

  def compute_source_term(self, y_nl, f, r, i, batch_size):
    k = self.compute_rate(r, f, batch_size)
    i = tf.squeeze(tf.where(i))
    if (tf.size(i) == 0):
      return k
    else:
      yi = tf.gather(y_nl, i, axis=-1)
      return k*tf.math.reduce_prod(yi, axis=-1, keepdims=True)

  def compute_rate(self, r, f, batch_size):
    k = self.rates_values[r]
    if self.use_corr:
      pass
      # corr = tf.gather(f, indices=self.rates_indices[0], axis=-2) \
      #      - tf.gather(f, indices=self.rates_indices[1], axis=-2)
      # corr = tf.math.reduce_sum(corr, axis=-1)
      # return tf.math.exp(corr)
    else:
      return tf.repeat(
        input=tf.reshape(k, (1,1)),
        repeats=batch_size,
        axis=0
      )

  def update_bvec(self, b, r, w, s_l):
    # Species indices and stoichiometric coefficients
    # > Reactants
    ir, cr = [self.rates_indices["reac"][k][r] for k in ("i", "c")]
    # > Products
    ip, cp = [self.rates_indices["prod"][k][r] for k in ("i", "c")]
    # Update 'b' vector
    for (sign, ix, cx) in (
      (-1.0, ir, cr),
      ( 1.0, ip, cp)
    ):
      i = tf.where(ix[s_l])
      c = sign*tf.gather(cx[s_l], i)
      b = tf.tensor_scatter_nd_add(
        tensor=b,
        indices=i,
        updates=c*tf.transpose(w)
      )
    return b

  def update_amat(self, a, r, w):
    # Species indices and stoichiometric coefficients
    # > Reactants
    ir, cr = [self.amat_indices["reac"][k][r] for k in ("i", "c")]
    # > Products
    ip, cp = [self.amat_indices["prod"][k][r] for k in ("i", "c")]
    # Update 'a' vector
    for (sign, ix, cx) in (
      (-1.0, ir, cr),
      ( 1.0, ip, cp)
    ):
      i = tf.where(ix)
      c = sign*tf.gather(cx, i)
      a = tf.tensor_scatter_nd_add(
        tensor=a,
        indices=i,
        updates=c*tf.transpose(w)
      )
    return a

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": self.rates,
      "use_corr": self.use_corr
    }
    base_config = super(POLLUOperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class POLLUSolUpdate(nn_elem.Layer):
  """Matrix exponential-based update for POLLU reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    use_corr=True,
    use_pade=False,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(POLLUSolUpdate, self).__init__(**kwargs)
    # POLLUOperators layer
    self.operators = POLLUOperators(
      rates=rates,
      use_corr=use_corr
    )
    # LinODE Layer
    self.linode = meno_lay.LinODE(
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
    t, nar_0, y_nl, f = inputs
    # Compute operators
    A = self.operators((y_nl, f), training=training)
    # Compute state-wise number densities
    nar = self.linode((t, nar_0, A), training=training)
    # Ensure conservation of Argon particles
    nar /= tf.math.reduce_sum(nar, axis=-1, keepdims=True)
    nar *= tf.math.reduce_sum(nar_0, axis=-1, keepdims=True)
    return nar

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.operators.rates),
      "use_corr": self.operators.use_corr,
      "use_pade": self.linode.use_pade,
      "use_expmv": self.linode.use_expmv,
      "max_degree": self.linode.max_degree,
      "max_nb_squarings": self.linode.max_nb_squarings,
      "expm_dtype": self.linode.expm_dtype,
      "add_inv_loss": self.linode.add_inv_loss,
      "inv_loss_weight": self.linode.inv_loss_weight
    }
    base_config = super(POLLUSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
