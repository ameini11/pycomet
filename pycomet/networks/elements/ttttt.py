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
class LinAutoencoder(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    phi,
    psi,
    xref,
    xscale,
    mask=None,
    use_log=False,
    **kwargs
  ):
    super(LinAutoencoder, self).__init__(**kwargs)
    # Basis
    self.phi = phi
    self.psi = psi
    # Reference and scaling values
    self.xref = xref
    self.xscale = xscale
    # Masking
    self.mask = mask
    # Log-transformation
    self.use_log = bool(use_log)

  # Building
  # ===================================
  def build(self, *args, **kwargs):
    """
    Construct the linear autoencoder by setting basis functions,
    projection operators, and scaling transformations.
    """
    # Load all the arrays
    for k in ("phi", "psi", "xref", "xscale", "mask"):
      x = getattr(self, k)
      x = self._load_array(k, x)
      setattr(self, k, x)
    # Ensure mask is boolean and properly shaped
    if (self.mask is None):
      self.mask = np.ones_like(self.xref)
    self.mask = self.mask.astype(bool).reshape(-1)
    # Dimensions
    self.size_x = len(self.mask)             # Full dimension
    self.size_xnot = int(np.sum(~self.mask)) # Number of excluded dimensions
    self.size_zhat = self.phi.shape[1]       # Number of reduced dimensions
    # Biorthogonalize the basis
    self.phi = self.phi @ np.linalg.inv(self.psi.T @ self.phi)
    # Construct full basis matrices
    self.phi = self._build_basis(self.phi)
    self.psi = self._build_basis(self.psi)
    # Handle reference and scaling values
    self.xref[~self.mask]= 0.0
    self.xscale[~self.mask]= 1.0
    self.xref = self.xref.reshape(1,-1)
    self.xscale = self.xscale.reshape(-1,1)
    # Build encoder and decoder
    self.encoder = self.psi / self.xscale
    self.decoder = (self.phi * self.xscale).T
    # Convert to tf.tensor
    for k in ("phi", "psi", "xref", "xscale", "mask"):
      x = getattr(self, k)
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
    basis = np.zeros((self.size_x, self.size_xnot + self.size_zhat))
    # Insert basis for projection
    basis[self.mask,:self.size_zhat] = pxi
    # Insert identity elements for not selected states
    ix = np.where(~self.mask)[0]
    iy = self.size_zhat + np.arange(self.size_xnot)
    basis[ix,iy] = 1.0
    return basis

  # Calling
  # ===================================
  def call(self, x, method=1, training=None):
    if (self.use_log and method in (4,)):
      utils.raise_err(
        ValueError,
        f"Method '{method}' is not valid when log transformation is used."
      )
    if (method == 1):
      return self._encode_vec(x)
    elif (method == 2):
      return self._decode_vec(x)
    elif (method == 3):
      return self._encode_decode_vec(x)
    elif (method == 4):
      return self._reduce_jac(x)

  def _encode_vec(self, x):
    # Transform
    if self.use_log:
      x = tf.math.log(x)
    x = x - self.xref
    # Encode
    return tf.linalg.matmul(x, self.encoder)

  def _decode_vec(self, z):
    # Decode
    x = tf.linalg.matmul(z, self.decoder)
    # Transform
    x = x + self.xref
    if self.use_log:
      x = tf.math.exp(x)
    return x

  def _encode_decode_vec(self, x):
    return self._decode_vec(self._encode_vec(x))

  def _reduce_jac(self, x):
    if self.use_log:
      utils.raise_err(
        ValueError,
        f"The input array '{name}' must be a `str` " \
        f"or a `np.ndarray` in '{self.name}'."
      )
    x = tf.linalg.matmul(x, self.decoder)
    x = tf.linalg.matmul(self.encoder, x)
    return x

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'phi': ops.to_numpy(self.phi).squeeze(),
      'psi': ops.to_numpy(self.psi).squeeze(),
      'xref': ops.to_numpy(self.xref).squeeze(),
      'xscale': ops.to_numpy(self.xscale).squeeze(),
      'use_log': self.use_log
    }
    base_config = super(LinAutoencoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  




  # Encoding/Decoding
  # ===================================
  def encode(
    self,
    x: np.ndarray,
    is_der: bool = False
  ) -> np.ndarray:
    """
    Encode the full state vector into the reduced-order representation.

    :param y: Full state vector of shape `(nb_eqs,)` or `(N, nb_eqs)`,
              where `N` is the batch size (optional).
    :type y: np.ndarray
    :param is_der: If True, encodes a derivative vector instead of a state
                   vector.
    :type is_der: bool

    :return: Encoded reduced-state representation of shape `(rom_dim,)`
             if `y` is `(nb_eqs,)`, or `(N, rom_dim)` if `y` is `(N, nb_eqs)`.
    :rtype: np.ndarray

    :raises ValueError: If ROM is not built or if input shape
                        mismatches `nb_eqs`.
    """
    if (not self.built):
      raise ValueError("ROM model not built.")
    # Ensure correct dimensions
    if (x.shape[-1] != self.nb_eqs):
      raise ValueError("Input state shape mismatch: expected last " \
                      f"dimension {self.nb_eqs}, got {x.shape[-1]}.")
    # Encode
    if (not is_der):
      x = x - self.xref
    return x @ self.encoder.T

  def decode(
    self,
    z: np.ndarray,
    is_der: bool = False
  ) -> np.ndarray:
    """
    Decode the reduced-order state back into full-state representation.

    :param z: Reduced state vector of shape `(rom_dim,)` or `(N, rom_dim)`,
              where `N` is the batch size (optional).
    :type z: np.ndarray
    :param is_der: If True, decodes a derivative vector instead of a state
                   vector.
    :type is_der: bool

    :return: Reconstructed full state vector of shape `(nb_eqs,)`
             if `z` is `(rom_dim,)`, or `(N, nb_eqs)` if `z` is `(N, rom_dim)`.
    :rtype: np.ndarray

    :raises ValueError: If ROM is not built or if input shape
                        mismatches `rom_dim`.
    """
    if (not self.built):
      raise ValueError("ROM model not built.")
    # Ensure correct dimensions
    if (z.shape[-1] != self.rom_dim):
      raise ValueError("Input reduced state shape mismatch: Expected " \
                      f"last dimension {self.rom_dim}, got {z.shape[-1]}.")
    # Decode
    x = z @ self.decoder.T
    if (not is_der):
      x = x + self.xref
    return x

  def reduce_jac(
    self,
    j: np.ndarray
  ) -> np.ndarray:
    """
    Perform encoding and decoding on a Jacobian matrix.

    :param j: Jacobian matrix of shape `(nb_eqs, nb_eqs)`.
    :type j: np.ndarray

    :return: Reduced Jacobian matrix.
    :rtype: np.ndarray
    """
    return self.encoder @ j @ self.decoder
