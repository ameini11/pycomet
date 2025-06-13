import copy
import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class StateCorrection(nn_elem.Layer):

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
    state_prop=None,
    **kwargs
  ):
    super(StateCorrection, self).__init__(**kwargs)
    # Static state-specific properties
    self.state_prop = self.read_prop(state_prop)
    # Blocks for encoding 'time' and 'state' spaces
    # > Hyperparameters
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout = dropout
    self.layer_norm = layer_norm
    # > Initialization
    self.blocks = {}
    for name in ("time", "state"):
      layer_size_i = copy.deepcopy(self.layer_size)
      if (name == "state"):
        layer_size_i[-1] += 2
      self.blocks[name] = nn_elem.FFBlock(
        layer_size=layer_size_i,
        activation=self.activation,
        kernel_initializer=self.kernel_initializer,
        regularizer=self.regularizer,
        layer_norm=self.layer_norm,
        dropout=self.dropout
      )
    # > Latent space
    self.latent = self.layer_size[-1]
    self.splits = [self.latent, 1, 1]

  def read_prop(self, prop):
    if isinstance(prop, (np.ndarray, tf.Tensor)):
      pass
    elif isinstance(prop, str):
      prop = np.loadtxt(prop)
    else:
      utils.raise_err(
        ValueError,
        "The input data must be either an array with shape [nb_states,:] " \
        "or a string representing a valid file path for reading."
      )
    prop = ops.to_numpy(prop).squeeze()
    # Check shape
    if (len(prop.shape) == 1):
      prop = prop.reshape(-1,1)
    return tf.constant(prop, dtype=bkd.get_floatx())

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Encode 'time' inputs > shape = [n,1,p]
    # -------------
    yt = self.blocks["time"](inputs, training=training)
    yt = tf.expand_dims(yt, axis=1)
    # Encode static state-specific properties > shape = [1,m,p+2]
    # -------------
    ys = self.blocks["state"](self.state_prop, training=training)
    ys = tf.expand_dims(ys, axis=0)
    # > Splitting
    ys, d, c = tf.split(ys, num_or_size_splits=self.splits, axis=-1)
    # > Scaling
    ys *= d
    # > Shifting
    c = tf.repeat(input=c, repeats=tf.shape(yt)[0], axis=0)
    # Linerly combine latent spaces
    # -------------
    return tf.concat([yt*ys,c], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "layer_size": self.layer_size,
      "activation": self.activation,
      "kernel_initializer": self.kernel_initializer,
      "regularizer": self.regularizer,
      "dropout": self.dropout,
      "layer_norm": self.layer_norm,
      "state_prop": ops.to_numpy(self.state_prop)
    }
    base_config = super(StateCorrection, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinPROM(nn_elem.LinAutoencoder):

  # Initialization
  # ===================================
  def __init__(
    self,
    phi,
    psi,
    xref=None,
    xscale=None,
    mask=None,
    z_dim=None,
    **kwargs
  ):
    super(LinPROM, self).__init__(
      phi=phi,
      psi=psi,
      xref=xref,
      xscale=xscale,
      mask=mask,
      z_dim=z_dim,
      use_log=False,
      **kwargs
    )

  def build(self, *args, **kwargs):
    super(LinPROM, self).build(*args, **kwargs)
    # Reduced tensors
    self.red_decoder = tf.math.reduce_sum(self.decoder, axis=-1)
    if self.use_xref:
      self.xref_norm_fac = 1.0 / (1.0 - tf.math.reduce_sum(self.xref))

  # Calling
  # ===================================
  def call(self, x, method=1, training=None):
    method = int(method)
    if ((method < 1) or (method > 6)):
      utils.raise_err(
        ValueError,
        f"Method '{method}' not valid."
      )
    if (self.use_log and (method > 3)):
      utils.raise_err(
        ValueError,
        f"Method '{method}' is not valid when the log transformation is used."
      )
    if (method == 1):
      return self._encode_vec(x)
    elif (method == 2):
      return self._decode_vec(x)
    elif (method == 3):
      return self._encode_decode_vec(x)
    elif (method == 4):
      return self._reduce_lin_ops(*x)
    elif (method == 5):
      return self._compute_rho(x)

  def _reduce_lin_ops(self, A, b=None):
    # Equation: dz/dt = Ar @ z + br
    # First order term
    # > Ar = E @ A @ D
    Ar = tf.linalg.matmul(A, tf.transpose(self.decoder))
    Ar = tf.linalg.matmul(tf.transpose(self.encoder), Ar)
    # Zeroth order term
    # > br = E @ (A @ xref + b)
    br = 0.0
    br_active = False
    if self.use_xref:
      br = br + tf.linalg.matvec(A, self.xref)
      br_active = True
    if (b is not None):
      br = br + b
      br_active = True
    br = tf.linalg.matmul(br, self.encoder) if br_active else None
    # Return reduced operator
    return Ar, br

  def _compute_rho(self, rhoz):
    # Equation: rho = sum(D @ rhoz) / (1 - sum(xref))
    rho = tf.linalg.matvec(rhoz, self.red_decoder)
    if self.use_xref:
      rho *= self.xref_norm_fac
    return rho
