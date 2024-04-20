"""
Action of the matrix exponential on a vector.
"""
import numpy as np
import scipy as sp
import pycomet as pcm
import tensorflow as tf


def expmv(A, B, deg=18):
  # Reshape
  is_B_2d = (tf.rank(B) == 2)
  if is_B_2d:
    B = tf.expand_dims(B, axis=-1)
  # Set polynomial degree
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

def _compute_l1_norm(A):
  l1_norm = tf.math.reduce_sum(tf.math.abs(A), axis=-2, keepdims=True)
  l1_norm = tf.math.reduce_max(l1_norm, axis=-1, keepdims=True)
  is_finite = tf.math.is_finite(tf.math.reduce_max(l1_norm))
  return l1_norm, is_finite

# Polynomial norms and degrees
# -------------------------------------
def _get_maxnorm(dtype, deg=None):
  if (dtype in (tf.float64, tf.complex128)):
    maxnorm = {
      1:  2.220446049250313e-16,
      2:  2.580956802971767e-08,
      4:  3.397168839976962e-04,
      8:  4.991228871115323e-02,
      12: 2.996158913811580e-01,
      18: 1.090863719290036e+00
    }
  elif (dtype in (tf.float16, tf.float32, tf.complex64)):
    maxnorm = {
      1:  1.192092800768788e-07,
      2:  5.978858893805233e-04,
      4:  5.116619363445086e-02,
      8:  5.800524627688768e-01,
      12: 1.461661507209034e+00,
      18: 3.010066362817634e+00
    }
  else:
    pcm.utils.raise_err(
      ValueError,
      f"'linalg.expm' does not support matrices of type '{dtype}'."
    )
  if (deg is not None):
    return maxnorm[deg]
  else:
    return maxnorm

# Taylor approximation
# -------------------------------------
def _ov_factorial(deg):
  f = []
  for i in range(2,deg+1):
    fi = 1.0 / sp.special.factorial(n=i, exact=True)
    f.append(fi)
  return tf.constant(np.array(f))
_f = _ov_factorial(deg=18)

def _taylor(A, B, deg):
  Ap_B = tf.matmul(A, B)
  B += Ap_B
  f = tf.cast(_f, dtype=B.dtype)
  for i in range(2,deg+1):
    Ap_B = tf.matmul(A, Ap_B)
    B += f[i-2] * Ap_B
  return B

# Scaling and squaring operations
# -------------------------------------
def _scaling(A, l1_norm, maxnorm):
  # Constant value
  two = tf.constant(2.0, l1_norm.dtype)
  # Compute scaling
  ov_log2 = 1/tf.math.log(two)
  squarings = ov_log2 * tf.math.log((1/maxnorm) * l1_norm)
  squarings = tf.math.maximum(tf.math.floor(squarings), 0)
  # Apply scaling
  As = (1 / tf.math.pow(two, squarings)) * A
  return As, squarings

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
