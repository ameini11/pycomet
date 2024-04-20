"""
Action of the matrix exponential on a vector.

See:
  - paper: https://doi.org/10.1137/100788860
  - code: https://github.com/higham/expmv
"""
import tensorflow as tf


def expmv(
  A,
  b,
  p=10
):
  yp = tf.linalg.matvec(A, b)
  y = b + yp
  for i in range(2,p+1):
    f = 1.0 / tf.math.reduce_prod(tf.range(1, i+1, dtype=b.dtype))
    yp = tf.linalg.matvec(A, yp)
    y += f * yp
  return y
