import numpy as np
import tensorflow as tf

from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.meno import layers as meno_lay
from silx.io.dictdump import h5todict
_CFG = get_config()

from .crmodel import CRModel
from .mixture import Mixture


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCROperators(nn_elem.Layer):
  """Compute the linear operators for plasma reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    processes,
    use_corr=False,
    scale_corr=False,
    species_order=("em", "Ar", "Arp"),
    **kwargs
  ):
    super(PlasmaCROperators, self).__init__(**kwargs)
    # Mixture
    mix = Mixture(
      species=species,
      species_order=tuple(species_order)
    )
    # Reaction rates
    self.rates = self.read_rates(rates)
    self.set_rates_values_indices()
    # Corrections
    self.use_corr = use_corr

  def read_rates(self, rates):
    if isinstance(rates, str):
      rates = h5todict(h5file=rates)
    rates = tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)
    rates["indices"] = rates["indices"].astype(np.int32)
    return rates

  def set_rates_values_indices(self):
    self.nb_comp = int(np.amax(self.rates["indices"]))+1
    # Set scatter update
    self.scatter_update = []
    i = self.rates["indices"]
    K = np.arange(self.nb_comp**2).reshape(self.nb_comp,self.nb_comp)
    for (rows, cols, sign, direction) in (
      (0, 0, -1, "fwd"),
      (0, 1,  1, "bwd"),
      (1, 0,  1, "fwd"),
      (1, 1, -1, "bwd")
    ):
      indices = K[i[rows],i[cols]].reshape(-1,1)
      data = {
        "sign": tf.constant(sign, dtype=bkd.get_floatx()),
        "direction": direction,
        "indices": tf.constant(indices, dtype="int32")
      }
      self.scatter_update.append(data)
    # Set rates values and indices
    self.rates_values = {}
    for direction in ("fwd", "bwd"):
      self.rates_values[direction] = tf.constant(
        self.rates[direction].reshape(1,-1), dtype=bkd.get_floatx()
      )
    self.rates_indices = tf.constant(self.rates["indices"], dtype="int32")

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > ne: electron number density
    # > f:  state-wise corrective factors
    ne, f = inputs
    # Correct reaction rates
    rates = tf.nest.map_structure(tf.identity, self.rates_values)
    if self.use_corr:
      corr = self.compute_correction(f)
      for direction in ("fwd", "bwd"):
        rates[direction] *= corr
    else:
      for direction in ("fwd", "bwd"):
        rates[direction] = tf.repeat(
          input=rates[direction],
          repeats=tf.shape(ne)[0],
          axis=0
        )
    # Compose linear operators: A from dydt=A@y
    return self.compose_lin_ops(rates, ne)

  def compute_correction(self, f):
    corr = tf.gather(f, indices=self.rates_indices[0], axis=-2) \
         - tf.gather(f, indices=self.rates_indices[1], axis=-2)
    corr = tf.math.reduce_sum(corr, axis=-1)
    return tf.math.exp(corr)

  def compose_lin_ops(self, rates, ne):
    A = tf.zeros((self.nb_comp**2, tf.shape(ne)[0]), dtype=bkd.get_floatx())
    for data in self.scatter_update:
      A = tf.tensor_scatter_nd_add(
        tensor=A,
        indices=data["indices"],
        updates=tf.transpose(data["sign"]*rates[data["direction"]])
      )
    A = ne * tf.transpose(A)
    A = tf.reshape(A, (-1,self.nb_comp,self.nb_comp))
    return A

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": self.rates,
      "use_corr": self.use_corr
    }
    base_config = super(PlasmaCROperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaSolUpdate(nn_elem.Layer):
  """Matrix exponential-based update for plasma reactor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    use_corr=True,
    epsilon=1e-12,
    use_pade=False,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(PlasmaSolUpdate, self).__init__(**kwargs)
    # PlasmaCROperators layer
    self.operators = PlasmaCROperators(
      rates=rates,
      use_corr=use_corr
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
    t, nar_0, ne, f = inputs
    # Compute operators
    A = self.operators((ne, f), training=training)
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
      "rates": self.operators.rates,
      "use_corr": self.operators.use_corr,
      "epsilon": self.linode.epsilon,
      "use_pade": self.linode.use_pade,
      "use_expmv": self.linode.use_expmv,
      "max_degree": self.linode.max_degree,
      "max_nb_squarings": self.linode.max_nb_squarings,
      "expm_dtype": self.linode.expm_dtype,
      "add_inv_loss": self.linode.add_inv_loss,
      "inv_loss_weight": self.linode.inv_loss_weight
    }
    base_config = super(PlasmaSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
