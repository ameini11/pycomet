import numpy as np
import tensorflow as tf

from .. import _NN_MOD
from pycomet import utils
from . import activations
from . import initializers
from . import regularizers
from . import layers as pcm_layers
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.config import get_config
from keras import layers as tf_layers
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MLP(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer='GlorotNormal',
    regularizer=None,
    layer_norm=None,
    dropout=None,
    **kwargs
  ):
    super(MLP, self).__init__(**kwargs)
    # Hyperparameters initialization
    # -----------------------
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.regularizer = regularizers.get(**regularizer)
    self.layer_norm = layer_norm
    self.dropout = dropout
    # Layers initialization
    # -----------------------
    self.lays = []
    # Dense
    self.lays.append(
      tf_layers.Dense(
        units=self.units,
        activation=self.activation,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.regularizer,
        bias_regularizer=self.regularizer,
        name=self.name+'_dense'
      )
    )
    # Normalization
    if self.layer_norm:
      self.lays.append(
        tf_layers.LayerNormalization(
          axis=-1,
          epsilon=1e-5,
          center=True,
          scale=True,
          beta_initializer='zeros',
          gamma_initializer='ones',
          beta_regularizer=self.regularizer,
          gamma_regularizer=self.regularizer,
          beta_constraint=None,
          gamma_constraint=None,
          name=self.name+'_layerNorm'
        )
      )
    # Dropout
    if self.dropout:
      self.lays.append(
        pcm_layers.Dropout(**dropout, name=self.name+'_dropout')
      )

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x
    for lay in self.lays:
      y = lay(y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'units': self.units,
      'activation': self.activation,
      'use_bias': self.use_bias,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'layer_norm': self.layer_norm,
      'dropout': self.dropout
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FFBlock(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    layer_norm=None,
    dropout=None,
    interpolating=None,
    **kwargs
  ):
    super(FFBlock, self).__init__(**kwargs)
    # Hyperparameters initialization
    # -----------------------
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.layer_norm = layer_norm
    self.dropout = dropout
    self.interpolating = interpolating
    # Layers initialization
    # -----------------------
    if (self.interpolating is not None):
      self.interpolating['name'] = self.name
      self.lays = [pcm_layers.Interpolating(**self.interpolating)]
    else:
      # Checking
      c1 = all([isinstance(k, (list, tuple)) for k in (activation, layer_size)])
      c2 = (len(activation) != len(layer_size)-1)
      if ((not c1) or c2):
        utils.raise_err(
          ValueError,
          f"'activation' and 'layer_size' inputs passed to {self.name} " \
          "block have to be a 'list'/'tuple' of lengths: " \
          "len(activation) = len(layer_size)-1"
        )
      # Hidden Layers
      names = [f'hidden_{i}' for i in range(len(self.layer_size)-2)]
      names = names + ['output']
      nb_mlp = len(self.layer_size)-1
      self.lays = [
        MLP(
          units=self.layer_size[i+1],
          activation=self.activation[i],
          kernel_initializer=self.kernel_initializer,
          regularizer=self.regularizer,
          layer_norm=self.layer_norm if (i == nb_mlp-1) else False,
          dropout=self.dropout,
          name=self.name+'_'+names[i]
        ) for i in range(nb_mlp)
      ]

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x
    for lay in self.lays:
      y = lay(y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'layer_size': self.layer_size,
      'activation': self.activation,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'layer_norm': self.layer_norm,
      'dropout': self.dropout,
      'interpolating': self.interpolating
    }
    base_config = super(FFBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class ResPIBlock(FFBlock):
  """An improved residual block architecture for PINNs.

  See:
    Sifan Wang, Yujun Teng, Paris Perdikaris
    'Understanding and mitigating gradient pathologies in physics-informed
    neural networks'
    https://arxiv.org/abs/2001.04536
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    uv_activation=['sigmoid','sigmoid'],
    kernel_initializer=None,
    regularizer=None,
    layer_norm=None,
    dropout=None,
    **kwargs
  ):
    super(ResPIBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Hyperparameters initialization
    # -----------------------
    for i in range(1, len(self.layer_size)-2):
      if (self.layer_size[i] != self.layer_size[i+1]):
        utils.raise_err(
          ValueError,
          "The number of units of each hidden layer " \
          f"must be the same in '{self.name}'."
        )
    self.uv_activation = uv_activation
    # Layers initialization
    # -----------------------
    self.u, self.v = [
      MLP(
        units=self.layer_size[1],
        activation=self.uv_activation[i],
        kernel_initializer=self.kernel_initializer,
        regularizer=self.regularizer,
        layer_norm=self.layer_norm,
        dropout=self.dropout,
        name=name
      ) for (i, name) in enumerate(('u', 'v'))
    ]

  # Calling
  # ===================================
  def call(self, x, training=None):
    # Input layer
    y = x
    # Encoder layers
    u = self.u(y, training=training)
    v = self.v(y, training=training)
    # Hidden layers
    y = self.lays[0](y, training=training)
    for i in range(1,len(self.layer_size)-2):
      z = self.lays[i](y, training=training)
      y = (1 - z) * u + z * v
    # Output layer
    y = self.lays[-1](y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'uv_activation': self.uv_activation
    }
    base_config = super(ResPIBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class LinAutoencoder(pcm_layers.Layer):

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     phi,
#     psi,
#     xref,
#     xscale,
#     use_log=False,
#     **kwargs
#   ):
#     super(LinAutoencoder, self).__init__(**kwargs)
#     # Load basis
#     self.phi = self.load_tensor(phi)
#     self.psi = self.load_tensor(psi)
#     self.phi = self.phi @ tf.linalg.inv(tf.transpose(self.psi) @ self.phi)
#     self.z_dim = int(self.phi.shape[1])
#     # Load reference and scaling values
#     self.xref = tf.reshape(self.load_tensor(xref), (1,-1))
#     self.xscale = tf.reshape(self.load_tensor(xscale), (-1,1))
#     # Log-transformation
#     self.use_log = bool(use_log)
#     # Build encoder and decoder
#     self.encoder = self.psi / self.xscale
#     self.decoder = tf.transpose(self.phi * self.xscale)

#   def load_tensor(self, x):
#     if isinstance(x, str):
#       x = np.loadtxt(x)
#     return tf.constant(x, dtype=bkd.get_floatx())

#   # Calling
#   # ===================================
#   def call(self, x, method=1, training=None):
#     if (method == 1):
#       return self.encode(x)
#     elif (method == 2):
#       return self.decode(x)
#     elif (method == 3):
#       return self.decode(self.encode(x))

#   def encode(self, x):
#     # Transform
#     if self.use_log:
#       x = tf.math.log(x)
#     x = x - self.xref
#     # Encode
#     return tf.linalg.matmul(x, self.encoder)

#   def decode(self, z):
#     # Decode
#     x = tf.linalg.matmul(z, self.decoder)
#     # Transform
#     x = x + self.xref
#     if self.use_log:
#       x = tf.math.exp(x)
#     return x

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       'phi': ops.to_numpy(self.phi).squeeze(),
#       'psi': ops.to_numpy(self.psi).squeeze(),
#       'xref': ops.to_numpy(self.xref).squeeze(),
#       'xscale': ops.to_numpy(self.xscale).squeeze(),
#       'use_log': self.use_log
#     }
#     base_config = super(LinAutoencoder, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinAutoencoder(pcm_layers.Layer):

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
    use_log=False,
    **kwargs
  ):
    super(LinAutoencoder, self).__init__(**kwargs)
    # Basis
    self.phi = phi
    self.psi = psi
    # Scaling values
    self.xref = xref
    self.use_xref = False if (xref is None) else True
    # Reference values
    self.xscale = xscale
    self.use_xscale = False if (xscale is None) else True
    # Masking
    self.mask = mask
    # Latent dimension
    self.z_dim = z_dim
    # Log-transformation
    self.use_log = bool(use_log)

  # Building
  # ===================================
  def build(self, *args, **kwargs):
    # Load all the arrays
    for k in ("phi", "psi", "xref", "xscale", "mask"):
      x = getattr(self, k)
      x = self._load_array(k, x)
      setattr(self, k, x)
    # Dimensions
    n, r = self.phi.shape
    if self.use_xref:
      n = len(self.xref)
    elif self.use_xscale:
      n = len(self.xscale)
    self.x_dim = n
    self.z_dim = r if (self.z_dim is None) else int(self.z_dim)
    # Masking
    if (self.mask is None):
      self.mask = np.ones(self.x_dim)
    self.mask = self.mask.astype(bool).reshape(-1)
    self.xnot_dim = int(np.sum(~self.mask)) # Number of excluded dimensions
    # Biorthogonalize the basis
    self.phi = self.phi[:,:self.z_dim]
    self.psi = self.psi[:,:self.z_dim]
    self.phi = self.phi @ np.linalg.inv(self.psi.T @ self.phi)
    # Construct full basis matrices
    self.phi = self._build_basis(self.phi)
    self.psi = self._build_basis(self.psi)
    # Handle reference and scaling values
    if self.use_xref:
      self.xref[~self.mask]= 0.0
    if self.use_xscale:
      self.xscale[~self.mask]= 1.0
    # Build encoder and decoder
    if self.use_xscale:
      self.encoder = (self.psi.T / self.xscale).T
      self.decoder = self.phi.T * self.xscale
    else:
      self.encoder = self.psi
      self.decoder = self.phi.T
    # Convert to tf.Tensor
    for k in ("phi", "psi", "xref", "xscale", "mask", "encoder", "decoder"):
      x = getattr(self, k)
      if (x is not None):
        x = tf.constant(x, dtype=bkd.get_floatx())
        setattr(self, k, x)

  def _load_array(self, name, x):
    if (x is not None):
      if (not isinstance(x, (str, np.ndarray))):
        utils.raise_err(
          ValueError,
          f"The input array '{name}' must be a `str` " \
          f"or a `np.ndarray` in '{self.name}'."
        )
      if isinstance(x, str):
        x = np.loadtxt(x)
      return x

  def _build_basis(self, pxi):
    """
    Constructs the full basis matrix, incorporating projection basis
    and identity elements for excluded states.

    :param pxi: Basis matrix (`phi` or `psi`).
    :type pxi: np.ndarray

    :return: Full basis matrix.
    :rtype: np.ndarray
    """
    # Allocate full basis matrix
    basis = np.zeros((self.x_dim, self.xnot_dim + self.z_dim))
    # Insert basis for projection
    basis[self.mask,:self.z_dim] = pxi
    # Insert identity elements for not selected states
    ix = np.where(~self.mask)[0]
    iy = self.z_dim + np.arange(self.xnot_dim)
    basis[ix,iy] = 1.0
    return basis

  # Calling
  # ===================================
  def call(self, x, method=1, training=None):
    method = int(method)
    if ((method < 1) or (method > 3)):
      utils.raise_err(ValueError, f"Method '{method}' not valid.")
    if (method == 1):
      return self._encode_vec(x)
    elif (method == 2):
      return self._decode_vec(x)
    elif (method == 3):
      return self._encode_decode_vec(x)

  def _encode_vec(self, x):
    # Transform
    if self.use_log:
      x = tf.math.log(x)
    if self.use_xref:
      x = x - self.xref
    # Encode
    z = tf.linalg.matmul(x, self.encoder)
    return z

  def _decode_vec(self, z):
    # Decode
    x = tf.linalg.matmul(z, self.decoder)
    # Transform
    if self.use_xref:
      x = x + self.xref
    if self.use_log:
      x = tf.math.exp(x)
    return x

  def _encode_decode_vec(self, x):
    return self._decode_vec(self._encode_vec(x))

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "z_dim": self.z_dim,
      "use_log": self.use_log
    }
    for k in ("phi", "psi", "xref", "xscale", "mask"):
      config[k] = ops.to_numpy(getattr(self, k))
    base_config = super(LinAutoencoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MixedLaysBlock(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    layers,
    **kwargs
  ):
    super(MixedLaysBlock, self).__init__(**kwargs)
    # Order layers by index
    indices = list(layers.keys())
    indices.sort()
    # Initialize layers
    self.layers = {i: self.get_lay(layers[i]) for i in indices}

  def get_lay(self, layer={}):
    if callable(layer):
      return layer
    else:
      lay_cls = utils.get_class(modules=_NN_MOD, name=layer['lay_id'])
      if ('kwargs' not in layer):
        layer['kwargs'] = {}
      return lay_cls(**layer['kwargs'])

  # Calling
  # ===================================
  def call(self, inputs, training=False, *args, **kwargs):
    y = inputs
    for layer in self.layers.values():
      y = layer(y, training=training, *args, **kwargs)
    return y

  def __call__(self, inputs, training=False, *args, **kwargs):
    return self.call(inputs, training=training, *args, **kwargs)

  def forward(self, inputs, training=False, *args, **kwargs):
    return self.call(inputs, training=training, *args, **kwargs)
