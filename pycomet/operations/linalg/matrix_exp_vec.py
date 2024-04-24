"""
Action of the matrix exponential on a vector.
"""
import numpy as np
import scipy as sp
import tensorflow as tf

from .matrix_exp import _compute_l1_norm, _expm_2by2, _get_maxnorm, _scaling


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
    # Set polynomial degree
    if (deg is None):
      deg = 18
    deg = min(deg, 18)
    # Compute L1 norm
    l1_norm, is_finite = _compute_l1_norm(A)
    if is_finite:
      # Max norm allowed by 'deg'
      maxnorm = _get_maxnorm(A.dtype, deg)
      # Scale matrix A
      As, squarings = _scaling(A, l1_norm, maxnorm)
      # Compute action of eA on B
      eA_B = _taylor(As, B, deg)
      eA_B = _squaring(As, eA_B, squarings, deg)
    else:
      eA_B = B
  # Reshape
  if is_B_2d:
    eA_B = tf.squeeze(eA_B, axis=-1)
  return eA_B

# Taylor approximation
# -------------------------------------
def _ov_factorial(deg):
  f = []
  for i in range(2,deg+1):
    fi = 1.0 / sp.special.factorial(n=i, exact=True)
    f.append(fi)
  return tf.constant(np.array(f), dtype="float64")
_f = _ov_factorial(deg=18)

def _taylor(A, B, deg):
  Ap_B = tf.matmul(A, B)
  B += Ap_B
  f = tf.cast(_f, dtype=B.dtype)
  for i in range(2,deg+1):
    Ap_B = tf.matmul(A, Ap_B)
    B += f[i-2] * Ap_B
  return B

# def expmv(A, B, s, m):
#   F = B
#   # Squarings
#   for i in tf.range(s, dtype=B.dtype):
#     # Polynomial
#     for j in tf.range(m, dtype=B.dtype):
#       B = (1.0 / (s * (j+1.0))) * tf.matmul(A, B)
#       F += B
#     B = F
#   return F

# Squaring operation
# -------------------------------------
def _squaring(A, B, squarings, deg):
  # Maximum number of squarings operation
  max_squarings = tf.math.reduce_max(squarings)
  if (max_squarings > 0):
    # Define stop condition: max squaring reached
    def cond(i, _):
      return tf.math.less(i, max_squarings)
    # Define squaring operation
    def body(i, Bi):
      return i+1, tf.where(
        condition=tf.math.less(i, squarings),
        x=_taylor(A, Bi, deg),
        y=Bi
      )
    # Perform squarings
    i = tf.constant(0.0, max_squarings.dtype)
    _, B = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=[i, B]
    )
  return B
