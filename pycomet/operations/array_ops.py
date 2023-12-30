import numpy as np
import pandas as pd
import tensorflow as tf

from pycomet import utils
from tensorflow.python.keras import backend


def convert_struct_to_tensor(x):
    return tf.nest.map_structure(
        lambda y: tf.convert_to_tensor(y, dtype=backend.floatx()), x
    )


def get_struct_shape(structure=None, axis=None):
    if structure is not None:
        if axis is not None:
            fn = lambda x: x.shape[axis] if x is not None else None
        else:
            fn = lambda x: x.shape if x is not None else None
        return tf.nest.map_structure(fn, structure)

def shuffle_struct_data(data):
    # Number of samples
    samples = tf.nest.map_structure(lambda x: x.shape[0], data)
    # Permute indices
    indices = {
        s: np.random.permutation(s) for s in np.unique(tf.nest.flatten(samples))
    }
    indices = tf.nest.map_structure(lambda x: indices[x], samples)
    # Shuffle data structure
    data = tf.nest.map_structure(
        lambda x, i: np.take(x, i, axis=0), data, indices
    )
    return data

def unpack_list(data, n_elem=3):
    if isinstance(data, (tuple)):
        data = list(data)
    if not isinstance(data, list):
        return ([data] + [None]*(n_elem-1))
    elif len(data) <= n_elem:
        return (data + [None]*(n_elem-len(data)))
    else:
        utils.raise_value_err(
            "The number of elements in the container "
            "has to be less or equal than 'n_elem'."
        )
    
def extract_data(container, indices=None, axis=1):
    if indices is not None:
        if isinstance(container, (list, tuple)):
            if not isinstance(indices, dict):
                utils.raise_value_err(
                    "With container as `list`/`tuple`, the indices "
                    "must be a `dict` with keys-values as integers."
                )
            k = list(indices.keys())[0]
            return extract_data(container[k], indices[k], axis)
        elif isinstance(container, (np.ndarray, tf.Tensor)):
            return extract_data_from_array(container, indices, axis)
        else:
            utils.raise_value_err(
                "Container not valid. Valid options: "
                "[`np.ndarray`, `tf.Tensor`] or `list`/`tuple` of them."
            )
    else:
        return container

def extract_data_from_array(array, indices=None, axis=1):
    if indices is None:
        if isinstance(array, pd.DataFrame):
            array = array.to_numpy()
        return array
    else:
        if not isinstance(indices, (list,tuple)):
            indices = [indices]
        if any([isinstance(c, (list,tuple)) for c in indices]):
            return [_extract(array, c, axis=axis) for c in indices]
        else:
            return _extract(array, indices, axis=axis)

def _extract(array, indices=[], axis=1):
    indices = tf.nest.flatten(indices)
    if isinstance(array, pd.DataFrame):
        if all([isinstance(c, str) for c in indices]):
            return array[indices].to_numpy()
        else:
            return np.take(array.to_numpy(), indices, axis=axis)
    elif isinstance(array, tf.Tensor):
        return tf.gather(array, indices, axis=axis)
    else:
        return np.take(array, indices, axis=axis)

# def extract_data_from_array(array, indices=None, axis=1):
#     if indices is None:
#         return array
#     else:
#         if not isinstance(indices, (list,tuple)):
#             indices = [indices]
#         if any([isinstance(c, (list,tuple)) for c in indices]):
#             return [np.take(array, c, axis=axis) for c in indices]
#         else:
#             return np.take(array, indices, axis=axis)

def remove_elem_from_struct(data, elem=None):
    if isinstance(data, dict):
        return {
            k: remove_elem_from_struct(v, elem) for k, v in data.items() \
                if k is not None and v is not None
        }
    elif isinstance(data, (list, tuple, set)):
        return type(data)([
            remove_elem_from_struct(i, elem) for i in data if i is not None
        ])
    else:
        return data

def reshape(x=None, dim=1, fmt='np'):
    if x is not None and 1 <= dim <= 3:
        if fmt == 'np':
            x = np.array(x)
            diff = int(dim - len(x.shape))
            if diff > 0:
                for _ in range(diff):
                    x = np.expand_dims(x,-1)
                return x
            elif diff == 0:
                return x
            else:
                for _ in range(-diff):
                    x = np.squeeze(x,-1)
                return x
        elif fmt == 'tf':
            x = tf.convert_to_tensor(x)
            diff = int(dim - len(x.shape))
            if diff > 0:
                for _ in range(diff):
                    x = tf.expand_dims(x, 0)
                return x
            elif diff == 0:
                return x
            else:
                for _ in range(-diff):
                    x = tf.squeeze(x, 0)
                return x
        else:
            utils.raise_value_err('Array format not recognized.')
    else:
        utils.raise_value_err('The input array is more than 3D.')






def squeeze_list(
  iterable
):
  """
  Squeeze a list by removing its outer list structure.

  :param iterable: An iterable (e.g., list, tuple) to be squeezed.
  :type iterable: list, tuple, or any iterable

  :return: The squeezed iterable if it contains only one element;
           otherwise, the original iterable is returned.
  :rtype: any

  This function takes an iterable and removes its outer list structure if it
  contains only one element. If the iterable contains multiple elements or is
  empty, it returns the original iterable.
  """
  iterable = list(iterable)
  if len(iterable) == 1:
    return iterable[-1]
  else:
    return iterable


def get_cardinality(
  x
):
  """
  Get the cardinality, meaning the number of points, of arrays or
  structured arrays.

  :param x: Input array or structured arrays.
  :type x: list, tuple, numpy.ndarray, or similar

  This function calculates the cardinality, which represents the number of
  points in an array or structured array. If the input is a list or tuple,
  it recursively calculates the cardinality for the first element. If it's a
  numpy array or similar, it returns the shape's first dimension (number of
  rows) as the cardinality.
  """
  return get_cardinality(x[0]) if isinstance(x, (list,tuple)) else x.shape[0]




def cast(structure, bkd="tf"):
  if (bkd == "tf"):
    fn = lambda x: tf.cast(x, get_floatx(bkd))
  else:
    fn = lambda x: x.astype(get_floatx(bkd))
  return tf.nest.map_structure(fn, structure)