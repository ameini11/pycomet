import numpy as np
import tensorflow as tf

from pycomet import utils
from tensorflow.python.keras import backend


def convert_struct_to_tensor(x):
    return tf.nest.map_structure(
        lambda y: tf.convert_to_tensor(y, dtype=backend.floatx()), x
    )

def get_cardinality(x):
    return get_cardinality(x[0]) if isinstance(x, (list,tuple)) else x.shape[0]

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

def extract_data_from_array(array, indices=None, axis=1):
    if indices is None:
        return array
    else:
        if not isinstance(indices, (list,tuple)):
            indices = [indices]
        if any([isinstance(c, (list,tuple)) for c in indices]):
            return [np.take(array, c, axis=axis) for c in indices]
        else:
            return np.take(array, indices, axis=axis)

def extract_data(container, indices=None, axis=1):
    if indices is not None:
        if not isinstance(indices, (dict, int)):
            utils.raise_value_err(
                "Indices not valid. It has to be a `int` or a "
                "`dict` with keys-values as integers."
            )
        if isinstance(container, (list, tuple)):
            k = list(indices.keys())[0]
            return extract_data(container[k], indices[k], axis)
        if isinstance(container, (np.ndarray, tf.Tensor)):
            return extract_data_from_array(container, indices, axis)
        else:
            utils.raise_value_err(
                "Container not valid. Valid options: "
                "[`np.ndarray`, `tf.Tensor`] or `list`/`tuple` of them."
            )
    else:
        return container

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