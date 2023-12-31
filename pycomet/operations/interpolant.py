import tensorflow as tf

from pycomet import utils
from pycomet import device as dev
_EPSILON = dev.get_epsilon()


# Interpolant
# ====================================
# See https://en.wikipedia.org/wiki/Polyharmonic_spline
def solve(c, f, phi, smoothing=0.0):
  # Get dimensions
  n, d = tf.shape(c)
  k = tf.shape(f)[-1]
  if (d is None):
    utils.raise_err(
      ValueError,
      "The dimensionality of the input points (d) must be " \
      "statically-inferrable."
    )
  if (k is None):
    utils.raise_err(
      ValueError,
      "The dimensionality of the output values (k) must be " \
      "statically-inferrable."
    )
  # Construct the linear system
  with tf.name_scope("construct_linear_system"):
    # > Matrix A
    amat = phi(_pairwise_squared_distance_matrix(c))
    if (smoothing > 0):
      amat += smoothing * tf.eye(n)
    # > Matrix B
    ones = tf.ones((n, 1))
    bmat = tf.concat([c, ones], axis=1)
    bmat_ncols = bmat.shape[1]
    # > Left hand side
    lhs_zeros = tf.zeros((bmat_ncols, bmat_ncols))
    block_right = tf.concat([bmat, lhs_zeros], axis=0)
    block_left = tf.concat([amat, tf.transpose(bmat)], axis=0)
    lhs = tf.concat([block_left, block_right], axis=1)
    # > Right hand side
    rhs_zeros = tf.zeros((d + 1, k))
    rhs = tf.concat([f, rhs_zeros], axis=0)
  # Solve the linear system
  with tf.name_scope("solve_linear_system"):
    w_v = tf.linalg.solve(lhs, rhs)
    w = w_v[:n,:]
    v = w_v[n:,:]
  return w, v

def apply(x, c, phi, w, v):
  # Compute the contribution from the rbf term
  d_phi = phi(_cross_squared_distance_matrix(x, c))
  rbf_term = tf.matmul(d_phi, w)
  # Compute the contribution from the linear term
  ones = tf.ones_like(x[...,:1])
  x_pad = tf.concat([x, ones], axis=1)
  linear_term = tf.matmul(x_pad, v)
  return rbf_term + linear_term
  
# Nonlinear kernel function
# ====================================
def get_phi(order):
  if (order == 1):
    return _phi_1
  elif (order == 2):
    return _phi_2
  elif (order == 4):
    return _phi_4
  elif (order % 2 == 0):
    return lambda r: _phi_even(r, order)
  else:
    return lambda r: _phi_odd(r, order)

def _phi_1(r):
  r_eps = tf.maximum(r, _EPSILON)
  return tf.sqrt(r_eps)

def _phi_2(r):
  r_eps = tf.maximum(r, _EPSILON)
  return 0.5 * r * tf.math.log(r_eps)

def _phi_4(r):
  r_eps = tf.maximum(r, _EPSILON)
  return 0.5 * tf.square(r) * tf.math.log(r_eps)

def _phi_even(r, order: int):
  r_eps = tf.maximum(r, _EPSILON)
  return 0.5 * tf.pow(r_eps, 0.5 * order) * tf.math.log(r_eps)

def _phi_odd(r, order: int):
  r_eps = tf.maximum(r, _EPSILON)
  return tf.pow(r_eps, 0.5 * order)

# Tensor operation - Distance matrix
# ====================================
def _cross_squared_distance_matrix(x, y):
  # Compute quadratic norm
  x_sq_norm = tf.reduce_sum(tf.square(x), axis=1)
  y_sq_norm = tf.reduce_sum(tf.square(y), axis=1)
  # Increase rank
  x_sq_norm = tf.expand_dims(x_sq_norm, axis=1)
  y_sq_norm = tf.expand_dims(y_sq_norm, axis=0)
  # Perform matrix multiplication
  x_yt = tf.matmul(x, y, adjoint_b=True)
  # Compute squared distance
  return x_sq_norm - 2 * x_yt + y_sq_norm

def _pairwise_squared_distance_matrix(x):
  # Compute quadratic values
  x_xt = tf.matmul(x, x, adjoint_b=True)
  # Extract batch diagonal
  x_xt_diag = tf.linalg.diag_part(x_xt)
  # Increase rank
  x_xt_diag = tf.expand_dims(x_xt_diag, axis=1)
  # Compute squared distance
  return x_xt_diag - 2 * x_xt + tf.transpose(x_xt_diag)

# Tensor rank
# ====================================
def to_rank(x, rank, maxrank=2):
  # Get tensor rank
  xrank = tf.rank(x)
  if (xrank > maxrank):
    utils.raise_err(
      ValueError,
      f"The rank of the input tensor must be <= {maxrank}."
    )
  # Reshape tensor
  if (rank < maxrank+1):
    d = int(rank - xrank)
    if (d > 0):
      for _ in range(d):
        x = tf.expand_dims(x, axis=-1)
    else:
      for _ in range(d):
        x = tf.squeeze(x, axis=-1)
    return x
  else:
    utils.raise_err(
      ValueError,
      f"The desired rank of the output tensor must be <= {maxrank}."
    )
