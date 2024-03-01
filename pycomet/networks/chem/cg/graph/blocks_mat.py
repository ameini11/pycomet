import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CorrFactor(nn_elem.FFBlock):
  """Predicts state/group-specific corrective factor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
    layer_norm=None,
    static_prop=None,
    features_log_xform=None,
    **kwargs
  ):
    super(CorrFactor, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    self.static_prop = gno.utils.read_attr(
      data=static_prop, identifier="static_prop"
    )
    self.nb_comp = self.static_prop.shape[-2]
    # Conditioning inputs log-transformation
    self.features_log_xform = features_log_xform
    if (self.features_log_xform is not None):
      self.log_lay = nn_elem.Log(**self.features_log_xform)

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = []
    # > Conditioning inputs (e.g., [t, na, T]])
    x = tf.expand_dims(inputs, axis=1)
    features.append(tf.repeat(x, repeats=self.nb_comp, axis=1))
    # > Static state/group-specific physical properties
    n = tf.shape(inputs)[0]
    features.append(tf.repeat(self.static_prop, repeats=n, axis=0))
    features = tf.concat(features, axis=-1)
    # Predict corrective factor
    if (self.features_log_xform is not None):
      features = self.log_lay(features)
    f = super(CorrFactor, self).call(features, training=training)
    return tf.squeeze(f, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop),
      "features_log_xform": self.features_log_xform
    }
    base_config = super(CorrFactor, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdEulerIsoME(nn_elem.Layer):
  """Forward Euler-type update for the isothermal Master equation."""

  # Initialization
  # ===================================
  def __init__(
    self,
    processes,
    rates,
    **kwargs
  ):
    super(FwdEulerIsoME, self).__init__(**kwargs)
    self.processes = processes
    self.rates = rates
    for (k, x) in self.rates.items():
      if (k not in ("kef", "keb", "kd", "kr")):
        utils.raise_err(ValueError, f"Rates not valid: {k}")
      else:
        if isinstance(x, str):
          x = np.loadtxt(x)
        x = np.expand_dims(ops.to_numpy(x).squeeze(), axis=0)
        self.rates[k] = tf.constant(x)

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    t, n, f = utils.unpack_list(inputs, nb_elem=3)
    f = f+1
    # Extract number densities
    na, nm = tf.split(n, num_or_size_splits=[1,n.shape[1]-1], axis=1)
    # Compose corrected reaction rates
    if ("exc" in self.processes):
      ke = self.rates["kef"] * f
      ke = ke + tf.transpose(self.rates["keb"] * f, perm=[0,2,1])
      ked = tf.linalg.diag(tf.math.reduce_sum(ke, axis=-1))
      ke = tf.transpose(ke, perm=[0,2,1]) - ked
    if ("dis" in self.processes):
      kd = self.rates["kd"] * f
      kr = self.rates["kr"] * f
    # Construct source terms
    w = 0.0
    if ("exc" in self.processes):
      w = w + tf.einsum("nij,nj->ni",ke,nm*na)
    if ("dis" in self.processes):
      w = w - kd*nm*na + kr*na**3
    # Return linear update
    return nm + t*w

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "processes": self.processes,
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates)
    }
    base_config = super(FwdEulerIsoME, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
