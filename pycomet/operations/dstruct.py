import numpy as np
import pandas as pd
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd


# Data structure
# ====================================
def is_struct_tensor(struct):
  fun = lambda x: isinstance(x, tf.Tensor)
  is_tensor = tf.nest.map_structure(fun, struct)
  return all(tf.nest.flatten(is_tensor))

def cast_struct(struct):
  if is_struct_tensor(struct):
    fun = lambda x: tf.cast(x, bkd.get_floatx(bkd="tf"))
  else:
    fun = lambda x: x.astype(bkd.get_floatx(bkd="np"))
  return tf.nest.map_structure(fun, struct)

def convert_struct_to_tensor(struct):
  fun = lambda x: tf.convert_to_tensor(x, dtype=bkd.get_floatx(bkd="tf"))
  return tf.nest.map_structure(fun, struct)

def get_struct_shape(struct=None, axis=None):
  if (struct is not None):
    if (axis is not None):
      fun = lambda x: x.shape[axis] if (x is not None) else None
    else:
      fun = lambda x: x.shape if (x is not None) else None
    return tf.nest.map_structure(fun, struct)

def shuffle_struct(struct):
  # Number of samples
  nb_samples = tf.nest.map_structure(lambda x: x.shape[0], struct)
  # Permute indices
  sizes = np.unique(tf.nest.flatten(nb_samples))
  indices = {s: np.random.permutation(s) for s in sizes}
  indices = tf.nest.map_structure(lambda x: indices[x], nb_samples)
  # Shuffle data struct
  if is_struct_tensor(struct):
    fun = lambda x, i: tf.gather(x, i, axis=0)
  else:
    fun = lambda x, i: np.take(x, i, axis=0)
  return tf.nest.map_structure(fun, struct, indices)

def remove_none_from_struct(struct):
  if isinstance(struct, dict):
    _struct = {}
    for (k, v) in struct.items():
      if ((k is not None) and (v is not None)):
        _struct[k] = remove_none_from_struct(v)
    return _struct
  elif isinstance(struct, (list, tuple, set)):
    _struct = []
    for x in struct:
      if (x is not None):
        _struct.append(remove_none_from_struct(x))
    return type(struct)(_struct)
  else:
    return struct

# Data extraction
# ====================================
def extract_data(container, indices=None, axis=1):
  if (indices is not None):
    if isinstance(container, (list, tuple)):
      if (not isinstance(indices, dict)):
        utils.raise_err(
          ValueError,
          "With container as 'list'/'tuple', the indices " \
          "must be a 'dict' with keys-values as integers."
        )
      k = list(indices.keys())[0]
      return extract_data(container[k], indices[k], axis)
    elif isinstance(container, (np.ndarray, tf.Tensor)):
      return extract_data_from_tensor(container, indices, axis)
    else:
      utils.raise_err(
        ValueError,
        "Container not valid. Valid options: " \
        "['np.ndarray', 'tf.Tensor'] or 'list'/'tuple' of them."
      )
  else:
    return container

def extract_data_from_tensor(tensor, indices=None, axis=1):
  if (indices is None):
    if isinstance(tensor, pd.DataFrame):
      tensor = tensor.to_numpy()
    return tensor
  else:
    if (not isinstance(indices, (list,tuple))):
      indices = [indices]
    if any([isinstance(c, (list,tuple)) for c in indices]):
      return [_extract(tensor, c, axis=axis) for c in indices]
    else:
      return _extract(tensor, indices, axis=axis)

def _extract(tensor, indices=[], axis=1):
  indices = tf.nest.flatten(indices)
  if isinstance(tensor, pd.DataFrame):
    if all([isinstance(c, str) for c in indices]):
      return tensor[indices].to_numpy()
    else:
      return np.take(tensor.to_numpy(), indices, axis=axis)
  elif isinstance(tensor, tf.Tensor):
    return tf.gather(tensor, indices, axis=axis)
  else:
    return np.take(tensor, indices, axis=axis)

def get_cardinality(x):
  """
  Get the cardinality, meaning the number of points, of arrays or
  structured arrays.

  :param x: Input tensor or structured arrays.
  :type x: list, tuple, numpy.ndarray, or similar

  This function calculates the cardinality, which represents the number of
  points in an tensor or structured tensor. If the input is a list or tuple,
  it recursively calculates the cardinality for the first element. If it's a
  numpy tensor or similar, it returns the shape's first dimension (number of
  rows) as the cardinality.
  """
  return get_cardinality(x[0]) if isinstance(x, (list,tuple)) else x.shape[0]
