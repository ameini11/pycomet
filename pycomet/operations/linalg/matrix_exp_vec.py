"""
Action of the matrix exponential on a vector.
"""
import numpy as np
import tensorflow as tf

from .matrix_exp import _expm_2by2, _ident_like


def expmv(A, B, deg=None):
  A = tf.convert_to_tensor(A)
  if (A.shape[-2:] == [0, 0]):
    return A
  # Reshape
  is_B_2d = (len(tf.shape(B)) == 2)
  if is_B_2d:
    B = tf.expand_dims(B, axis=-1)
  if (A.shape[-1] == 2):
    # Analytical expression for a 2x2 matrix
    eA_B = tf.matmul(_expm_2by2(A), B)
  else:
    # Compute action of eA on B
    eA_B = _expmv(A, B)
  # Reshape
  if is_B_2d:
    eA_B = tf.squeeze(eA_B, axis=-1)
  return eA_B

# Polynomial norms and degrees
# -------------------------------------
def _get_thetas():
  # This table helps to compute bounds.
  # They seem to have been difficult to calculate, involving symbolic
  # manipulation of equations, followed by numerical root finding.
  _thetas = {
    # The first 30 values are from table A.3 of Computing Matrix Functions.
    1: 2.29e-16,
    2: 2.58e-8,
    3: 1.39e-5,
    4: 3.40e-4,
    5: 2.40e-3,
    6: 9.07e-3,
    7: 2.38e-2,
    8: 5.00e-2,
    9: 8.96e-2,
    10: 1.44e-1,
    # 11
    11: 2.14e-1,
    12: 3.00e-1,
    13: 4.00e-1,
    14: 5.14e-1,
    15: 6.41e-1,
    16: 7.81e-1,
    17: 9.31e-1,
    18: 1.09,
    19: 1.26,
    20: 1.44,
    # 21
    21: 1.62,
    22: 1.82,
    23: 2.01,
    24: 2.22,
    25: 2.43,
    26: 2.64,
    27: 2.86,
    28: 3.08,
    29: 3.31,
    30: 3.54,
    # The rest are from table 3.1 of
    # Computing the Action of the Matrix Exponential.
    35: 4.7,
    40: 6.0,
    45: 7.2,
    50: 8.5,
    55: 9.9
  }
  k = np.array(list(_thetas.keys()), dtype=np.float64)
  v = np.array(list(_thetas.values()), dtype=np.float64)
  return tf.constant(np.vstack([k, v]))

_thetas = _get_thetas()

# Core computation
# -------------------------------------
def _expmv(A, B):
  n = A.shape[-1]
  mu = (1.0/n) * tf.linalg.trace(A)
  A = A - mu * _ident_like(A)
  m, s = _get_m_s(A)
  return _loop_over_s_m(A, B, mu, s, m)

def _get_m_s(A):
  thetas = tf.cast(_thetas, dtype=A.dtype)
  # Compute L1 norm
  norm = tf.norm(A, ord=1, axis=[1,2])
  # Get theta index
  index = tf.math.argmin(
    thetas[0:1] * tf.math.ceil(
      tf.reshape(norm, (-1,1)) / thetas[1:2]
    ),
    axis=-1
  )
  # Get poly order
  m = tf.gather(thetas[0], indices=index, axis=0)
  # Get squarings
  theta = tf.gather(thetas[1], indices=index, axis=0)
  s = tf.math.ceil(norm / theta)
  return m, s

def _loop_over_s_m(A, B, mu, s, m):
  # Factor
  eta = tf.math.exp(mu/s)
  # Maximum number of iterations
  s_max = tf.math.reduce_max(s)
  # Define stop condition
  def cond(i, BFi):
    return tf.math.less(i, s_max)
  # Define operation
  def body(i, BFi):
    return i+1, tf.where(
      condition=tf.math.less(i, s),
      x=_loop_over_m(A, BFi, m, s, eta),
      y=BFi
    )
  # Perform loop
  i = tf.constant(0.0, s_max.dtype)
  _, BF = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=[i, tf.stack([B, B], axis=-1)]
  )
  return tf.gather(BF, indices=-1, axis=-1)

def _loop_over_m(A, BF, m, s, eta):
  # Initialize
  B, F = [tf.gather(BF, indices=i, axis=-1) for i in (0,1)]
  # Maximum number of iterations
  m_max = tf.math.reduce_max(m)
  # Define stop condition
  def cond(i, Bi, Fi):
    return tf.math.less_equal(i, m_max)
  # Define operation
  def body(i, Bi, Fi):
    ci = tf.math.less_equal(i, m)
    Bi = tf.where(
      condition=ci,
      x=(1.0/(s*i)) * tf.matmul(A, Bi),
      y=Bi
    )
    Fi = tf.where(
      condition=ci,
      x=Fi+Bi,
      y=Fi
    )
    return i+1, Bi, Fi
  # Perform loop
  i = tf.constant(1.0, m_max.dtype)
  _, B, F = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=[i, B, F]
  )
  return tf.stack([F, eta*F], axis=-1)
