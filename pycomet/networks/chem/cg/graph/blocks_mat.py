import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pycomet import utils
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


def _compose_feat(inputs, static_inputs, nb_elem):
  # Compose input features
  features = []
  # > Conditioning inputs (e.g., [t, na, T]])
  x = tf.expand_dims(inputs, axis=1)
  features.append(tf.repeat(x, repeats=nb_elem, axis=1))
  # > Static inputs
  n = tf.shape(inputs)[0]
  features.append(tf.repeat(static_inputs, repeats=n, axis=0))
  features = tf.concat(features, axis=-1)
  return features


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PFCorr(nn_elem.FFBlock):

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
    super(PFCorr, self).__init__(
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
    features = _compose_feat(inputs, self.static_prop, self.nb_comp)
    # Predict corrective factor
    if (self.features_log_xform is not None):
      features = self.log_lay(features)
    f = super(PFCorr, self).call(features, training=training)
    f = tf.squeeze(f, axis=-1)
    return (f,)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop),
      "features_log_xform": self.features_log_xform
    }
    base_config = super(PFCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdExcRateCorr(nn_elem.FFBlock):

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
    super(FwdExcRateCorr, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    if isinstance(static_prop, str):
      static_prop = np.loadtxt(static_prop)
    self.static_prop = ops.to_numpy(static_prop).squeeze()
    # Number of states/groups
    nb_comp, np_prop = self.static_prop.shape
    # Forward rates
    # > Create tensor
    fwd_rates = np.zeros((nb_comp,nb_comp,np_prop*2))
    # > Get ij lower triangular indices only
    i, j = np.tril_indices(nb_comp, k=-1)
    # > Fill tensor with ij properties
    pi = self.static_prop[i]
    pj = self.static_prop[j]
    fwd_rates[i,j,:] = np.hstack([pi, pj])
    # > Convert to tensorflow object
    fwd_rates = tf.constant(fwd_rates[1:,:-1])
    fwd_rates = tf.transpose(fwd_rates, perm=[2,0,1])
    fwd_rates_vec = tfp.math.fill_triangular_inverse(fwd_rates)
    fwd_rates_vec = tf.transpose(fwd_rates_vec)
    self.fwd_rates_vec = tf.expand_dims(fwd_rates_vec, axis=0)
    self.nb_fwd_rates = self.fwd_rates_vec.shape[1]
    # > Create padding operator
    self.q = tf.constant(np.eye(nb_comp)[:nb_comp-1])
    # Conditioning inputs log-transformation
    self.features_log_xform = features_log_xform
    if (self.features_log_xform is not None):
      self.log_lay = nn_elem.Log(**self.features_log_xform)

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = _compose_feat(inputs, self.fwd_rates_vec, self.nb_fwd_rates)
    # Predict corrective factor
    if (self.features_log_xform is not None):
      features = self.log_lay(features)
    f = super(FwdExcRateCorr, self).call(features, training=training)
    f = self.make_tril(f)
    return f

  def make_tril(self, f):
    """Make lower triangular."""
    f = tf.squeeze(f, axis=-1)
    # Fill lower triangular (diagonal included)
    f = tfp.math.fill_triangular(f)
    # Padding last row/column
    f = tf.einsum("...ik,...ij,...jl->...kl", self.q, f, self.q)
    # Rolling last row up
    f = tf.roll(f, shift=[1], axis=[1])
    return f

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop),
      "features_log_xform": self.features_log_xform
    }
    base_config = super(FwdExcRateCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdDisRateCorr(nn_elem.FFBlock):

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
    super(FwdDisRateCorr, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    if isinstance(static_prop, str):
      static_prop = np.loadtxt(static_prop)
    self.static_prop = ops.to_numpy(static_prop).squeeze()
    # Forward rates
    fwd_rates_vec = tf.constant(self.static_prop)
    self.fwd_rates_vec = tf.expand_dims(fwd_rates_vec, axis=0)
    self.nb_fwd_rates = self.fwd_rates_vec.shape[1]
    # Conditioning inputs log-transformation
    self.features_log_xform = features_log_xform
    if (self.features_log_xform is not None):
      self.log_lay = nn_elem.Log(**self.features_log_xform)

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = _compose_feat(inputs, self.fwd_rates_vec, self.nb_fwd_rates)
    # Predict corrective factor
    if (self.features_log_xform is not None):
      features = self.log_lay(features)
    f = super(FwdDisRateCorr, self).call(features, training=training)
    return f

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop),
      "features_log_xform": self.features_log_xform
    }
    base_config = super(FwdDisRateCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FwdRateCorr(object):

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
    processes=("exc",),
    static_prop=None,
    features_log_xform=None,
    **kwargs
  ):
    self.processes = processes
    self.lays = {}
    for (k, lay) in (("exc", FwdExcRateCorr), ("dis", FwdDisRateCorr)):
      if (k in self.processes):
        self.lays[k] = lay(
          layer_size=layer_size[k],
          activation=activation[k],
          kernel_initializer=kernel_initializer,
          regularizer=regularizer,
          dropout=dropout,
          layer_norm=layer_norm,
          static_prop=static_prop,
          features_log_xform=features_log_xform[k],
          **kwargs
        )

  # Calling
  # ===================================
  def __call__(self, inputs, training=False):
    return self.call(inputs, training=training)

  def call(self, inputs, training=False):
    f = []
    for k in ("exc", "dis"):
      if (k in self.processes):
        f.append(self.lays[k](inputs, training=training))
    return tuple(f)


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdEulerIsoME(nn_elem.Layer):
  """Forward Euler-type update for the isothermal Master equation."""

  # Initialization
  # ===================================
  def __init__(
    self,
    processes,
    rates,
    use_pfcorr=False,
    **kwargs
  ):
    super(FwdEulerIsoME, self).__init__(**kwargs)
    self.processes = processes
    self.rates = rates
    self.use_pfcorr = use_pfcorr
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
    t, n0, f = utils.unpack_list(inputs, nb_elem=3)
    # Correct reaction rates
    rates = self.correct_rates(f)
    # Extract number densities
    na, nm = tf.split(n0, num_or_size_splits=[1,n0.shape[1]-1], axis=1)
    # Construct source terms
    w = 0.0
    if ("exc" in self.processes):
      ke = rates["kef"] + tf.transpose(rates["keb"], perm=[0,2,1])
      ked = tf.linalg.diag(tf.math.reduce_sum(ke, axis=-1))
      ke = tf.transpose(ke, perm=[0,2,1]) - ked
      w = w + tf.einsum("nij,nj->ni",ke,nm*na)
    if ("dis" in self.processes):
      w = w - rates["kd"]*nm*na + rates["kr"]*na**3
    # Return linear update
    return nm + t * w

  def correct_rates(self, f):
    _rates = {}
    if ("exc" in self.processes):
      fe = f[0]
      if self.use_pfcorr:
        fe = tf.math.exp(tf.einsum("ni,nj->nij", fe, -fe))
      for k in ("kef", "keb"):
        _rates[k] = self.rates[k] * fe
    if ("dis" in self.processes):
      fd = tf.math.exp(f[0]) if self.use_pfcorr else f[1]
      for k in ("kd", "kr"):
        _rates[k] = self.rates[k] * fd
    return _rates

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "processes": self.processes,
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_pfcorr": self.use_pfcorr
    }
    base_config = super(FwdEulerIsoME, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
