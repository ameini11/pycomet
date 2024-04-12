import numpy as np
import tensorflow as tf


def pinv(
  a,
  rcond=None,
  validate_args=False,
  name=None
):
  if (rcond is not None):
    dtype = a.dtype.as_numpy_dtype
    shape = tf.shape(a)
    num_rows, num_cols = shape[-2], shape[-1]
    if (isinstance(num_rows, int) and isinstance(num_cols, int)):
      max_rows_cols = float(max(num_rows, num_cols))
    else:
      max_rows_cols = tf.cast(tf.math.maximum(num_rows, num_cols), dtype)
    rcond = tf.math.maximum(
      tf.cast(rcond, dtype),
      10.0 * max_rows_cols * np.finfo(dtype).eps
    )
  return tf.linalg.pinv(
    a,
    rcond=rcond,
    validate_args=validate_args,
    name=name
  )
