import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd
from pycomet.config import get_config
from pycomet.operations import linalg
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Eigen(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    use_pinv=False,
    inv_scaling=None,
    **kwargs
  ):
    super(Eigen, self).__init__(**kwargs)
    self.rcond = 1e-10
    self.use_pinv = use_pinv
    self.inv_scaling = inv_scaling

  # Properties
  # ===================================
  @property
  def v(self):
    return self._v

  @v.setter
  def v(self, value):
    self._v = value

  @property
  def vinv(self):
    return self._vinv

  @vinv.setter
  def vinv(self, value=None):
    if (value is None):
      _v = self.v
      if (self.inv_scaling is not None):
        _v = _v * self.inv_scaling
      if self.use_pinv:
        _vinv = linalg.pinv(_v, rcond=self.rcond)
      else:
        _vinv = tf.linalg.inv(_v)
      if (self.inv_scaling is not None):
        _vinv *= self.inv_scaling
      self._vinv = _vinv
    else:
      self._vinv = value

  # Calling
  # ===================================
  def call(self, inputs, mode="decompose", training=None):
    if (mode == "decompose"):
      return self.decompose(inputs)
    else:
      return self.reconstruct(inputs)

  def decompose(self, x):
    l, v = tf.linalg.eig(x)
    self.v, self.vinv = v, None
    return tf.expand_dims(l, axis=-1)

  def reconstruct(self, x):
    # > f(x) = V @ diag(x) @ V^(-1)
    # > Shapes: v=[n,i,j], x=[n,j,1], vinv=[n,j,i]
    v = tf.cast(self.v, dtype=x.dtype)
    vinv = tf.cast(self.vinv, dtype=x.dtype)
    return tf.matmul(v, x*vinv)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pinv": self.use_pinv,
      "inv_scaling": self.inv_scaling
    }
    base_config = super(Eigen, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MatrixExp(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    use_pade=True,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    **kwargs
  ):
    super(MatrixExp, self).__init__(**kwargs)
    self.use_pade = use_pade
    self.use_expmv = use_expmv
    self.max_degree = max_degree
    self.max_nb_squarings = max_nb_squarings
    self.expm_dtype = expm_dtype

  # Calling
  # ===================================
  def call(self, A, B=None, training=None):
    A, B = [self.recast(x, dtype=self.expm_dtype) for x in (A, B)]
    if (B is None):
      y = self.compute_expm(A)
    else:
      if self.use_expmv:
        y = linalg.expmv(A, B, deg=self.max_degree)
      else:
        y = tf.matmul(self.compute_expm(A), B)
    return self.recast(y, dtype=bkd.get_floatx())

  def recast(self, x, dtype=None):
    if (x is not None):
      if (dtype is not None):
        return tf.cast(x, dtype=dtype)
      else:
        return x

  def compute_expm(self, A):
    return linalg.expm(
      A=A,
      use_pade=self.use_pade,
      max_degree=self.max_degree,
      max_nb_squarings=self.max_nb_squarings
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pade": self.use_pade,
      "use_expmv": self.use_expmv,
      "max_degree": self.max_degree,
      "max_nb_squarings": self.max_nb_squarings,
      "expm_dtype": self.expm_dtype
    }
    base_config = super(MatrixExp, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinODE(nn_elem.Layer):
  """
  Integral solution of linear ODE with constant coefficients:
    > dydt = a @ y + b
    > y(t) = exp(a*t) @ y0 + (exp(a*t)-I) @ a^(-1) @ b
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    # Matrix exponential
    use_pade=True,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    # Non-invertibility loss
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(LinODE, self).__init__(**kwargs)
    # Matrix exponential layer
    self.use_pade = use_pade
    self.use_expmv = use_expmv
    self.max_degree = max_degree
    self.max_nb_squarings = max_nb_squarings
    self.expm_dtype = expm_dtype
    self.mat_exp = MatrixExp(
      use_pade=use_pade,
      use_expmv=use_expmv,
      max_degree=max_degree,
      max_nb_squarings=max_nb_squarings,
      expm_dtype=expm_dtype
    )
    # Additional loss to avoid non-invertibility of the matrix
    self.add_inv_loss = add_inv_loss
    self.inv_loss_weight = inv_loss_weight

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Manipulate inputs
    t, y0, a, b = utils.unpack_list(inputs, nb_elem=4)
    t, y0 = [tf.expand_dims(z, axis=-1) for z in (t, y0)]
    if (b is not None):
      b = tf.expand_dims(b, axis=-1)
    # Additional loss to avoid non-invertibility of the matrix
    if (training and self.add_inv_loss):
      self.compute_inv_loss(a)
    # Compute solution
    if (b is None):
      y = self.mat_exp(a*t, y0)
    else:
      y = tf.linalg.solve(a, b)
      y = self.mat_exp(a*t, y0+y) - y
    return tf.squeeze(y, axis=-1)

  def compute_inv_loss(self, a):
    # Compute singular values
    s = tf.linalg.svd(a, compute_uv=False)
    # Compute condition number
    smin = tf.math.reduce_min(s, axis=-1) + bkd.get_epsilon()
    smax = tf.math.reduce_max(s, axis=-1)
    cond = smax / smin
    # Minimize condition number
    loss = tf.math.reduce_mean(tf.math.log(cond))
    # Add loss
    self.add_loss(self.inv_loss_weight * loss)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_pade": self.use_pade,
      "use_expmv": self.use_expmv,
      "max_degree": self.max_degree,
      "max_nb_squarings": self.max_nb_squarings,
      "expm_dtype": self.expm_dtype,
      "add_inv_loss": self.add_inv_loss,
      "inv_loss_weight": self.inv_loss_weight
    }
    base_config = super(LinODE, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
